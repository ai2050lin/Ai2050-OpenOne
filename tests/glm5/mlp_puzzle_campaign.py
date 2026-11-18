"""Native B28--35 puzzle campaign: frozen factorial prediction and zero-ablation.

Author-constructed binding/translation materials, never synthetic model outputs.
Use: --stage freeze | pilot | discovery | select | calibrate | holdout | causal | report.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
import argparse, hashlib, json, math, random, shutil, subprocess, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from shared.dist_protocol.mechanism_trace import GpuLease,unbits
from shared.dist_protocol.natural_register_runner import snapshot
import mlp_condition_design as D
import mlp_condition_stream_load as SL
import mlp_condition_run as R

ROOT=D.ROOT; MODEL=D.MODEL; SNAPSHOT=D.SNAPSHOT
OUT=ROOT/'tests/glm5/result/mlp_puzzle_v1'
RUN='MLP-NATIVE-PUZZLES-20261109-v1'; SEED=202611090279
BLOCKS=list(range(28,36)); NS={'pilot':16,'discovery':128,'holdout':512}
sys.stdout.reconfigure(encoding='utf-8')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''): h.update(b)
    return h.hexdigest()
def save(name,d):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(d,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x).encode('utf-8')
    t=p.with_suffix(p.suffix+'.tmp');t.write_bytes(b);t.replace(p)
def bits(x): return x.detach().contiguous().view(torch.uint16).cpu().numpy().copy()
def log(stage,n,total,start,**extra):
    sec=time.monotonic()-start
    d={'time':datetime.now().astimezone().isoformat(),'stage':stage,'done':n,'total':total,
       'seconds':round(sec,1),'eta_seconds':round(sec*(total-n)/n,1) if n else None,**extra}
    if torch.cuda.is_initialized():
        d.update(allocated_MiB=round(torch.cuda.memory_allocated()/2**20),peak_MiB=round(torch.cuda.max_memory_allocated()/2**20))
        try:
            cmd="(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | Where-Object {$_.InstanceName -like 'pid_"+str(os.getpid())+"_*'} | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
            p=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,timeout=8)
            d['shared_gpu_MiB']=round(float(p.stdout.strip())/2**20,2) if p.returncode==0 and p.stdout.strip() else None
        except Exception:d['shared_gpu_MiB']=None
    print(json.dumps(d,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d)+'\n')

def features(r):
    q,c,l=r['query'],r['record'],r['language']
    return np.array([1,q,c,l,q*l,c*l,q*c,q*c*l],np.float64)

FRAMES={
 'pilot':['Record: {a} is {x}; {b} is {y}.'],
 'discovery':['Record: {a} is {x}; {b} is {y}.','The label on {a} says {x}. The label on {b} says {y}.',
              'Inventory entry: {a}, color {x}; {b}, color {y}.','The inspector recorded {a} as {x} and {b} as {y}.'],
 'holdout':['The color assigned to {a} is {x}, whereas {y} is the color assigned to {b}.',
            'Consult the two labels: {x} belongs to {a}; {y} belongs to {b}.',
            'In the final description, {a} has the {x} coating, and {b} has the {y} coating.',
            'The signed form pairs {a} with {x} and pairs {b} with {y}.']}
NOUNS={'pilot':['box','parcel'],'discovery':['box','parcel','folder','bag'],
       'holdout':['crate','envelope','container','packet','tray','case','bottle','basket']}

def material(tok,split,f):
    rng=random.Random(SEED+f+{'pilot':0,'discovery':10000,'holdout':20000}[split])
    noun=NOUNS[split][f%len(NOUNS[split])];base=10000 if split=='holdout' else 1000
    a=f'{noun} {base+2*f}';b=f'{noun} {base+2*f+1}'
    frame=FRAMES[split][f%len(FRAMES[split])]
    notes=['The room has a wooden shelf.','The clerk checked the date.','The record is stored in a cabinet.',
           'A visitor left a short note.','The door was closed at noon.','The table stands near the window.']
    background=' '.join(rng.sample(notes,3));rows=[]
    for q in (-1,1):
        for c in (-1,1):
            for l in (-1,1):
                x,y=('red','blue') if c==1 else ('blue','red')
                record=frame.format(a=a,b=b,x=x,y=y)
                query=a if q==1 else b
                question=(f'Which color belongs to {query}?' if split!='holdout' else f'Identify the color specified for {query}.')
                language=('English','Chinese') if l==1 else ('Chinese','English')
                instruction=(f'Answer in {language[0]}, not {language[1]}. Give only the color word.' if split!='holdout'
                             else f'Use {language[0]} for your answer; {language[1]} is excluded. Write just the color.')
                text=f'File {split[0]}{f:05d}. Use the supplied description, not typical object colors.\n{background}\n{record}\n{question}\n{instruction}'
                prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
                ids=tok.encode(prompt,add_special_tokens=False)
                positions=[len(tok.encode(prompt[:prompt.index(record)+len(record)],add_special_tokens=False))-1,
                           len(tok.encode(prompt[:prompt.index(question)+len(question)],add_special_tokens=False))-1,len(ids)-1]
                rows.append({'id':f'{split}-{f:04d}-{q}{c}{l}','family':f,'frame':f%len(FRAMES[split]),
                    'query':q,'record':c,'language':l,'semantic':q*c,'gold':('red' if q*c==1 else 'blue'),
                    'prompt':prompt,'tokens':ids,'positions':positions})
    return rows

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'plan.json').exists()
    assert shutil.disk_usage(ROOT).free>2.8*2**30
    start=time.monotonic();log('freeze',0,1,start)
    assert snapshot(MODEL)==SNAPSHOT
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    # Full alternative spellings are recorded; predictions are for the first color token.
    variants={'en_red':[' red','red'],'en_blue':[' blue','blue'],'zh_red':['红',' 红'],'zh_blue':['蓝',' 蓝']}
    ids={k:sorted(set(tok.encode(s,add_special_tokens=False)[0] for s in v)) for k,v in variants.items()}
    materials={}
    for split,n in NS.items():
        rows=[r for f in range(n) for r in material(tok,split,f)]
        save(f'{split}-materials.json',rows);materials[split]={'sha256':sha(OUT/f'{split}-materials.json'),'families':n,'rows':len(rows)}
    source_dirs=[ROOT/'tests/glm5/result/mlp_tail_b28_b31_v1',ROOT/'tests/glm5/result/mlp_tail_goodfire_v1']
    prior=np.load(ROOT/'tests/glm5/result/mlp_tail_relative_encoding_v1/per_unit.npz')
    reps=[];sources={};rng=np.random.default_rng(SEED)
    for b in BLOCKS:
        folder=source_dirs[b>=32];p=folder/f'B{b}-profile.json';profile=read(p);sources[str(p.relative_to(ROOT))]=sha(p)
        assert profile['model_snapshot']==SNAPSHOT
        std=np.array(profile['fields']['A']['std']);mx=np.maximum(np.abs(profile['fields']['A']['max']),np.abs(profile['fields']['A']['min']))
        scores={'token_predictable':-prior[f'B{b}'][:,1],'double_negative':prior[f'B{b}'][:,4],
                'rare_amplitude':mx/(std+1e-6),'high_variance':std}
        used=set()
        for kind,v in scores.items():
            for u in np.argsort(-v):
                if int(u) not in used:break
            used.add(int(u));reps.append({'block':b,'unit':int(u),'kind':kind,'archive_score':float(v[u])})
        for u in rng.choice([j for j in range(9728) if j not in used],2,replace=False):reps.append({'block':b,'unit':int(u),'kind':'random'})
    sources['tests/glm5/result/mlp_tail_relative_encoding_v1/per_unit.npz']=sha(ROOT/'tests/glm5/result/mlp_tail_relative_encoding_v1/per_unit.npz')
    cap=read(D.CAP);cap=[r for r in cap if r['gate_eligible']];rng.shuffle(cap);cap=cap[:512]
    save('capability-materials.json',cap)
    git=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    plan={'schema':'native_puzzle_plan.v1','kind':'real','run_id':RUN,'created_at':datetime.now().astimezone().isoformat(),
      'model_snapshot':SNAPSHOT,'model_path':str(MODEL),'seed':SEED,'code_sha256':sha(__file__),'git_sha':git,
      'materials':materials,'choice_ids':ids,'answer_variants':variants,'representatives':reps,'archive_sources':sources,
      'provenance':{'git_sha':git,'model_path':str(MODEL),'model_revision':SNAPSHOT,'seed':SEED,'tool':RUN},
      'design':{'factors':['query entity sign','record color assignment sign','requested language sign'],
        'features':['1','q','r','L','qL','rL','qr','qrL'],'null_columns':[0,1,2,3,4,5],
        'legal_information':'q,r,L parsed from authored input structure; no observed activation or future answer used as predictor input; qr rule is supplied, not discovered',
        'heldout':'new record IDs, new entity nouns, four unseen frames; eight variants stay with each family',
        'scope':'controlled record binding and requested output language; not parametric factual knowledge or unrestricted translation',
        'sampling':'48 archive-stratified representatives; 8-unit content and language groups per block selected only on discovery; matched same-block random groups',
        'regression':'orthogonal balanced factorial coefficients; train first96 discovery families, selection validation last32; refit128 after choosing',
        'behavior_gate':{'forced_choice_per_frame':.90,'free_top1_color_mean':.80},
        'response_gate':{'relative_mse_max':.80,'family_win_min':.65,'bootstrap_family_lower_gain':0,'alpha':.05/16},
        'effect_gate':{'absolute_mean_margin_change_min':.20,'prediction_mse_ratio_max':.80,'matched_excess_ci_lower':0,'alpha':.05/4},
        'interventions':'last-position native A zero ablation at doses .5 and1; native random group and equal-realized-residual-norm random direction; frozen stratum-average logit-change predictor',
        'causal_groups':['best content group','best language group','strongest unit of chosen content group','B33/U1521 rare prior'],
        'causal_families':256,'calibration_families':32,'capability_n':512,'capability_scope':'deterministic sample of exposed historical multiple-choice pool, descriptive side-effect check',
        'power':'512 family clusters for response,256 for effects; 16 response and4 causal primary tests. With unit-SD paired family differences, normal-approx MDE at80% power is approximately .17 and .21 SD respectively; clustered-template generalization not guaranteed.'},
      'resources':{'gpu':'one CUDA Qwen3-4B BF16 SDPA','max_stage_minutes':60,'disk_budget_GiB':2.6,
        'retention':'lossless BF16 full A for B28..35;all37 hidden states at evidence-end and final;discovery full final G/V;holdout G/V for frozen groups+representatives;full discovery logits;holdout full-vocabulary metrics computed before discard, reproducible from input/snapshot/code',
        'preflight_revision':'first freeze stopped before material creation because only3.4GiB free;storage reduced before any model observation'},
      'limits':['Old archive exposed','four heldout template families;record IDs are not independent semantic domains','one physical node','no training trajectory','first-token constrained answer;free generation separately audited']}
    save('plan.json',plan);log('freeze',1,1,start,representatives=len(reps),choice_ids=ids)

def checked():
    p=read(OUT/'plan.json');assert p['code_sha256']==sha(__file__),'Frozen source changed'
    for split,v in p['materials'].items():assert sha(OUT/f'{split}-materials.json')==v['sha256']
    return p
def load_model(plan):
    D.OUT=OUT;D.RUN=RUN;D.checked=lambda:plan
    def loader_save(name,body,_):save(name,body)
    R.save=loader_save
    return SL.load(MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)

class Capture:
    def __init__(self,m):
        self.m=m;self.data={};self.args={};self.handles=[];self.on=False
        for b in BLOCKS:
            layer=m.model.layers[b]
            for f,mod in [('G',layer.mlp.gate_proj),('V',layer.mlp.up_proj)]:
                self.handles.append(mod.register_forward_hook(self.vector(b,f)))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self.pre(b,'A')))
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.pre(b,'R')))
            self.handles.append(layer.post_attention_layernorm.register_forward_hook(self.vector(b,'Z')))
            self.handles.append(layer.register_forward_hook(self.vector(b,'H')))
            self.handles.append(layer.register_forward_pre_hook(self.kwargs(b),with_kwargs=True))
    def vector(self,b,f):
        def hook(mod,args,out):
            if self.on:self.data[f'{f}{b}']=out
        return hook
    def pre(self,b,f):
        def hook(mod,args):
            if self.on:self.data[f'{f}{b}']=args[0]
        return hook
    def kwargs(self,b):
        def hook(mod,args,kwargs):
            if self.on:self.args[b]=(args[1:],kwargs.copy())
        return hook
    def forward(self,rows,tok,hidden=False):
        self.on=True;self.data={};self.args={}
        batch=tok.pad({'input_ids':[r['tokens'] for r in rows]},return_tensors='pt').to('cuda')
        result=self.m(**batch,use_cache=False,output_hidden_states=hidden,logits_to_keep=1)
        self.on=False
        return result,self.data.copy(),batch
    def suffix(self,base,group,dose,random_delta=None):
        b=group['block'];a=base[f'A{b}'].clone();units=group['units']
        a[:,-1,units]=(a[:,-1,units].float()*(1-dose)).to(a.dtype)
        state=base[f'H{b}'].clone()
        state[:,-1]=base[f'R{b}'][:,-1]+self.m.model.layers[b].mlp.down_proj(a)[:,-1]
        if random_delta is not None:state[:,-1]=base[f'H{b}'][:,-1]+random_delta
        delta=state[:,-1].float()-base[f'H{b}'][:,-1].float()
        for layer in range(b+1,36):
            args,kwargs=self.args[layer];state=self.m.model.layers[layer](state,*args,**kwargs)
        logits=self.m.lm_head(self.m.model.norm(state)[:,-1:])[:,0]
        return logits,delta
    def close(self):
        for h in self.handles:h.remove()

def score(logits,row,p):
    d=p['choice_ids'];keys=['en_red','en_blue','zh_red','zh_blue'];v=[float(logits[d[k]].max()) for k in keys]
    correct=(0 if row['semantic']==1 else 1)+(0 if row['language']==1 else 2)
    other=(1 if row['semantic']==1 else 0)+(0 if row['language']==1 else 2)
    wrong_language=correct+2 if correct<2 else correct-2
    return {'scores':v,'gold':correct,'top1':int(logits.argmax()),'forced_correct':v[correct]>v[other],
            'top1_correct':int(logits.argmax()) in d[keys[correct]],'content_margin':v[correct]-v[other],
            'language_margin':v[correct]-v[wrong_language]}

def collect(m,tok,p,split):
    if split=='discovery':assert read(OUT/'pilot-behavior.json')['passed']
    if split=='holdout':assert (OUT/'frozen.json').exists() and read(OUT/'discovery-behavior.json')['passed']
    rows=read(OUT/f'{split}-materials.json');cap=Capture(m);start=time.monotonic();log(split,0,len(rows)//8,start);scores=[]
    frozen=read(OUT/'frozen.json') if split=='holdout' else None
    predictor=np.load(OUT/'predictors.npz')['logits'] if split=='holdout' else None
    try:
        for f in range(len(rows)//8):
            path=OUT/f'{split}-{f:04d}.npz';mp=path.with_suffix('.json');rs=rows[f*8:f*8+8]
            if mp.exists():scores.extend(read(mp)['scores']);continue
            res,base,batch=cap.forward(rs,tok,hidden=split!='pilot');lo=res.logits[:,0]
            ss=[{**score(lo[i],r,p),'family':f,'frame':r['frame'],'id':r['id']} for i,r in enumerate(rs)]
            arrays={}
            if split!='pilot':
                for b in BLOCKS:
                    arrays[f'A{b}']=bits(base[f'A{b}'][:,-1])
                    units=sorted(set([r['unit'] for r in p['representatives'] if r['block']==b]+[u for g in (frozen['groups'] if frozen else []) if g['block']==b for u in g['units']+g['controls']]))
                    for field in ['G','V']:
                        arrays[f'{field}{b}']=bits(base[f'{field}{b}'][:,-1] if split=='discovery' else base[f'{field}{b}'][:,-1,units])
                    arrays[f'GV_units{b}']=np.arange(9728) if split=='discovery' else np.array(units)
                    # Hook identity catches collecting G instead of the actual down-projection input.
                    expected=torch.nn.functional.silu(base[f'G{b}'][:,-1])*base[f'V{b}'][:,-1]
                    assert torch.equal(expected,base[f'A{b}'][:,-1]),('A identity',b)
                positions=torch.tensor([[r['positions'][0],r['positions'][-1]] for r in rs],device='cuda')+batch.input_ids.shape[1]-torch.tensor([len(r['tokens']) for r in rs],device='cuda')[:,None]
                ix=torch.arange(len(rs),device='cuda')[:,None]
                arrays['hidden']=bits(torch.stack([h[ix,positions] for h in res.hidden_states],1))
                if split=='discovery':arrays['logits']=bits(lo)
                if split=='holdout':
                    xx=np.stack([features(r) for r in rs]);lp=lo.double().log_softmax(-1)
                    pred=torch.tensor(xx@predictor,device='cuda').log_softmax(-1)
                    null=torch.tensor(xx[:,:6]@predictor[:6],device='cuda').log_softmax(-1)
                    kl=(lp.exp()*(lp-pred)).sum(-1).tolist();kl0=(lp.exp()*(lp-null)).sum(-1).tolist()
                    for s,k,k0 in zip(ss,kl,kl0):s.update(distribution_KL=k,distribution_baseline_KL=k0)
                np.savez_compressed(path,**arrays)
                assert shutil.disk_usage(ROOT).free>256*2**20,'Disk reserve exhausted'
            save(mp.name,{'plan_sha256':sha(OUT/'plan.json'),'scores':ss,'sha256':sha(path) if path.exists() else None})
            scores.extend(ss)
            if (f+1)%16==0 or f+1==len(rows)//8:log(split,f+1,len(rows)//8,start)
            assert time.monotonic()-start<3600
        frames=sorted(set(s['frame'] for s in scores));by={str(f):float(np.mean([s['forced_correct'] for s in scores if s['frame']==f])) for f in frames}
        top=float(np.mean([s['top1_correct'] for s in scores]));passed=min(by.values())>=.90 and top>=.80
        save(f'{split}-behavior.json',{'passed':passed,'forced_by_frame':by,'first_top1':top,'n':len(scores),'scope':'forced color selection plus open-vocabulary first-token check;full generation separately measured'})
        log(split+'_complete',len(scores),len(scores),start,passed=passed,first_top1=top,forced_by_frame=by)
    finally:cap.close()

def matrices(split,field,b=None):
    key=f'{field}{b}' if b is not None else field
    return np.concatenate([unbits(np.load(OUT/f'{split}-{f:04d}.npz')[key]) for f in range(NS[split])])

def select(p):
    assert read(OUT/'discovery-behavior.json')['passed'];assert not list(OUT.glob('holdout-[0-9]*.npz'))
    rows=read(OUT/'discovery-materials.json');X=np.stack([features(r) for r in rows]);train=96*8
    coeffs={};groups=[];start=time.monotonic();rng=np.random.default_rng(SEED);val=[]
    for b in BLOCKS:
        Y=matrices('discovery','A',b);coef=X[:train].T@Y[:train]/train;full=X.T@Y/len(X)
        scale=Y[:train].std(0)+1e-5;coeffs[f'A{b}']=full.astype(np.float32)
        for kind,col in [('binding',6),('language',3)]:
            # Select on train only; choose one block using the disjoint discovery validation families.
            rank=np.argsort(-np.abs(coef[col])/scale);units=rank[:8].tolist();available=set(range(9728))-set(units)
            control=[]
            for u in units:
                cand=np.array(sorted(available));dist=np.abs(np.log(scale[cand]/scale[u]));near=cand[np.argsort(dist)[:100]]
                j=int(rng.choice(near));control.append(j);available.remove(j)
            pred=X[train:]@coef[:,units];null=X[train:,:6]@coef[:6,units]
            mse=float(np.mean((Y[train:,units]-pred)**2));baseline=float(np.mean((Y[train:,units]-null)**2))
            # Language is explicitly a main effect; its test excludes only its L-dependent columns.
            if kind=='language':null=X[train:,[0,1,2,6]]@coef[[0,1,2,6]][:,units];baseline=float(np.mean((Y[train:,units]-null)**2))
            g={'id':f'{kind}-B{b}','block':b,'units':units,'controls':control,'kind':kind,'validation_ratio':mse/max(baseline,1e-12),
               'hypothesis':('query×record binding beyond additive query/record cues' if kind=='binding' else 'requested output language conditional response;not unrestricted translation')}
            groups.append(g);val.append({'group':g['id'],'mse':mse,'baseline':baseline})
        log('select',b-27,8,start)
    chosen=[min([g for g in groups if g['kind']==k],key=lambda g:g['validation_ratio']) for k in ['binding','language']]
    single={**chosen[0],'id':'binding-single','units':[chosen[0]['units'][0]],'controls':[chosen[0]['controls'][0]]}
    rare={'id':'rare-prior','kind':'rare','block':33,'units':[1521],'controls':[int(rng.integers(9728))],
          'hypothesis':'historical rare-amplitude association may not carry color binding'}
    # Output distribution predictor uses only the eight frozen prefix-derived factorial features.
    logits=np.zeros((8,151936),np.float64)
    for f in range(NS['discovery']):
        y=unbits(np.load(OUT/f'discovery-{f:04d}.npz')['logits']);logits+=X[f*8:f*8+8].T@y
    coeffs['logits']=(logits/len(X)).astype(np.float32)
    np.savez(OUT/'predictors.npz',**coeffs)
    save('frozen.json',{'plan_sha256':sha(OUT/'plan.json'),'predictors_sha256':sha(OUT/'predictors.npz'),'groups':groups,
        'causal_groups':[*chosen,single,rare],'validation':val,'representatives':p['representatives'],
        'frozen_at':datetime.now().astimezone().isoformat(),'holdout_unopened':True})

def metrics(logits,original,row,p,pred=None):
    s=score(logits,row,p);orig=original.double();edit=logits.double();lp=orig.log_softmax(-1);lq=edit.log_softmax(-1)
    s['KL_change']=float((lq.exp()*(lq-lp)).sum())
    if pred is not None:
        lr=pred.double().log_softmax(-1);s['KL_prediction']=float((lq.exp()*(lq-lr)).sum())
        ps=score(pred,row,p);bs=score(original,row,p)
        for key in ['content_margin','language_margin']:
            s['predicted_'+key]=ps[key]
            s[key+'_prediction_se']=(s[key]-ps[key])**2
            s[key+'_nochange_se']=(s[key]-bs[key])**2
    return s

def calibrate(m,tok,p):
    frozen=read(OUT/'frozen.json');groups=frozen['causal_groups'];rows=read(OUT/'discovery-materials.json');cap=Capture(m)
    sums=np.zeros((4,4,151936),np.float64);counts=np.zeros((4,4),int);start=time.monotonic();log('calibrate',0,32,start)
    try:
        for f in range(32):
            row=rows[f*8+(f%8)];res,base,_=cap.forward([row],tok);original=res.logits[0,0];st=(row['semantic']==1)*2+(row['language']==1)
            for j,g in enumerate(groups):
                lo,delta=cap.suffix(base,g,1);diff=(lo[0].float()-original.float()).cpu().numpy();diff-=diff.mean()
                sums[j,st]+=diff;counts[j,st]+=1
                zero,_=cap.suffix(base,g,0);assert torch.equal(zero[0],original),('zero replay',f,j)
            if (f+1)%8==0:log('calibrate',f+1,32,start)
        assert counts.min()>0
        np.savez(OUT/'effect-predictor.npz',changes=(sums/counts[:,:,None]).astype(np.float32),counts=counts)
        save('effect-frozen.json',{'frozen_sha256':sha(OUT/'frozen.json'),'effect_sha256':sha(OUT/'effect-predictor.npz'),
           'frozen_at':datetime.now().astimezone().isoformat(),'holdout_unopened':not list(OUT.glob('holdout-[0-9]*.npz'))})
    finally:cap.close()

def causal(m,tok,p):
    fr=read(OUT/'frozen.json');ef=read(OUT/'effect-frozen.json');assert ef['effect_sha256']==sha(OUT/'effect-predictor.npz')
    groups=fr['causal_groups'];coef=np.load(OUT/'effect-predictor.npz')['changes'];rows=read(OUT/'holdout-materials.json');cap=Capture(m)
    start=time.monotonic();log('causal',0,256,start);rng=torch.Generator(device='cuda').manual_seed(SEED)
    try:
        for f in range(256):
            path=OUT/f'causal-{f:04d}.json'
            if path.exists():continue
            row=rows[f*8+(f%8)];res,base,_=cap.forward([row],tok);original=res.logits[0,0];st=(row['semantic']==1)*2+(row['language']==1)
            # Prediction commitment precedes interventions on this family.
            save(f'commit-{f:04d}.json',{'sample_id':row['id'],'effect_frozen_sha256':sha(OUT/'effect-frozen.json'),'stratum':st,'doses':[.5,1.]})
            records=[]
            for j,g in enumerate(groups):
                for dose in [.5,1.]:
                    lo,delta=cap.suffix(base,g,dose);pred=original.float()+dose*torch.tensor(coef[j,st],device='cuda')
                    result=metrics(lo[0],original,row,p,pred);norm=float(delta.norm())
                    # Compare actual representable BF16 residual writes, not nominal input amplitude.
                    h=base[f'H{g["block"]}'][:,-1];noise=torch.randn(h.shape,device='cuda',generator=rng);noise/=noise.norm()
                    amount=norm
                    best=None
                    for _ in range(12):
                        proposal=(h.float()+noise*amount).to(h.dtype);actual=proposal.float()-h.float();got=float(actual.norm())
                        error=abs(got-norm)/max(norm,1e-12)
                        if best is None or error<best[0]:best=(error,actual.to(h.dtype),got)
                        if error<=.02:break
                        amount*=norm/max(got,1e-12)
                    error,nd,got=best;null,_=cap.suffix(base,g,0,random_delta=nd)
                    rand,_=cap.suffix(base,{**g,'units':g['controls']},dose)
                    records.append({'group':g['id'],'dose':dose,'edited':result,'norm_control':metrics(null[0],original,row,p),
                        'random_group':metrics(rand[0],original,row,p),'write_norm':norm,'control_norm':got,'norm_relative_error':error,
                        'norm_valid':bool(error<=.02),'units':g['units'],'block':g['block']})
                zero,_=cap.suffix(base,g,0);assert torch.equal(zero[0],original)
            save(path.name,{'sample':row,'baseline':score(original,row,p),'records':records,'restored_bitwise':True,
                           'commit_sha256':sha(OUT/f'commit-{f:04d}.json')})
            if (f+1)%16==0:log('causal',f+1,256,start)
            assert time.monotonic()-start<3600
    finally:cap.close()

def capability(m,tok,p):
    groups=read(OUT/'frozen.json')['causal_groups'][:2];items=read(OUT/'capability-materials.json');scores=[];start=time.monotonic()
    choices=[tok.encode(' '+k,add_special_tokens=False)[0] for k in 'ABCDE'];log('capability',0,len(items),start)
    for offset in range(0,len(items),8):
        chunk=items[offset:offset+8];prompts=[tok.apply_chat_template([{'role':'user','content':r['question']+'\n'+'\n'.join(f'{c}. {v}' for c,v in zip('ABCDE',r['choices']))+'\n请选择一个正确选项字母。'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:' for r in chunk]
        batch=tok(prompts,return_tensors='pt',padding=True).to('cuda');out={}
        for g in [None,*groups]:
            def edit(mod,args):
                a=args[0].clone();a[:,-1,g['units']]=0;return (a,)
            handle=m.model.layers[g['block']].mlp.down_proj.register_forward_pre_hook(edit) if g else None
            try:out[g['id'] if g else 'base']=m(**batch,use_cache=False,logits_to_keep=1).logits[:,0,choices].float().cpu().numpy()
            finally:
                if handle:handle.remove()
        for i,r in enumerate(chunk):scores.append({'id':r['id'],'gold':'ABCDE'.index(r['answer']),'scores':{k:v[i,:len(r['choices'])].tolist() for k,v in out.items()}})
        if (offset+len(chunk))%64==0:log('capability',offset+len(chunk),len(items),start)
    save('capability.json',{'rows':scores,'scope':'512 items from an exposed historical pool; descriptive bounded side effects'})

def generation(m,tok,p):
    rows=read(OUT/'holdout-materials.json');out=[];start=time.monotonic();log('generation',0,64,start)
    for f in range(64):
        r=rows[f*8+(f%8)];ids=torch.tensor([r['tokens']],device='cuda');generated=m.generate(ids,max_new_tokens=12,do_sample=False,pad_token_id=tok.pad_token_id)
        answer=tok.decode(generated[0,len(r['tokens']):],skip_special_tokens=True).strip()
        targets=['red'] if r['semantic']==1 else ['blue']
        if r['language']==-1:targets=['红'] if r['semantic']==1 else ['蓝']
        out.append({'id':r['id'],'answer':answer,'accepted':any(t in answer.lower() for t in targets),'scope':'substring content/language screening;raw answer retained'})
        if (f+1)%16==0:log('generation',f+1,64,start)
    save('generation.json',out)

def bootstrap_gain(errors,baseline,alpha=.003125):
    difference=baseline-errors;rng=np.random.default_rng(SEED);draws=np.mean(difference[rng.integers(len(difference),size=(4000,len(difference)))],1)
    return [float(v) for v in np.quantile(draws,[alpha/2,1-alpha/2])]

def report(p):
    frozen=read(OUT/'frozen.json');coef=np.load(OUT/'predictors.npz');rows=read(OUT/'holdout-materials.json');X=np.stack([features(r) for r in rows]);stats=[];reps=[];start=time.monotonic()
    for b in BLOCKS:
        Y=matrices('holdout','A',b);pred=X@coef[f'A{b}'];null=X[:,:6]@coef[f'A{b}'][:6]
        for g in [g for g in frozen['groups'] if g['block']==b]:
            units=g['units'];n=(null if g['kind']=='binding' else X[:,[0,1,2,6]]@coef[f'A{b}'][[0,1,2,6]])
            e=((Y[:,units]-pred[:,units])**2).reshape(-1,8,8).mean((1,2));e0=((Y[:,units]-n[:,units])**2).reshape(-1,8,8).mean((1,2))
            ci=bootstrap_gain(e,e0);ratio=float(e.mean()/max(e0.mean(),1e-12));wins=float(np.mean(e<e0))
            control=g['controls'];ce=((Y[:,control]-pred[:,control])**2).mean();cn=((Y[:,control]-n[:,control])**2).mean()
            worst=np.argsort(e-e0)[-3:][::-1];examples=[{'family':int(i),'error':float(e[i]),'baseline':float(e0[i]),'sample':rows[int(i)*8]} for i in worst]
            stats.append({**g,'relative_mse':ratio,'family_win_rate':wins,'gain_ci':ci,'random_group_relative_mse':float(ce/max(cn,1e-12)),
                 'passed':bool(ratio<=.8 and wins>=.65 and ci[0]>0),'counterexamples':examples})
        for r in [r for r in p['representatives'] if r['block']==b]:
            u=r['unit'];e=((Y[:,u]-pred[:,u])**2).reshape(-1,8).mean(1);e0=((Y[:,u]-null[:,u])**2).reshape(-1,8).mean(1)
            reps.append({**r,'relative_mse':float(e.mean()/max(e0.mean(),1e-12)),'family_win_rate':float(np.mean(e<e0)),
                'holdout_std':float(Y[:,u].std()),'sign_positive':float(np.mean(Y[:,u]>0)),'scope':'descriptive representative;not a separately multiplicity-confirmed semantic claim'})
        log('report',b-27,8,start)
    causals=[read(OUT/f'causal-{f:04d}.json') for f in range(256)];effects=[]
    for g in frozen['causal_groups']:
        field='language_margin' if g['kind']=='language' else 'content_margin'
        for dose in [.5,1.]:
            rec=[(f,next(r for r in f['records'] if r['group']==g['id'] and r['dose']==dose)) for f in causals]
            losses=np.array([f['baseline'][field]-r['edited'][field] for f,r in rec]);control=np.array([f['baseline'][field]-r['norm_control'][field] for f,r in rec]);valid=np.array([r['norm_valid'] for f,r in rec]);basekl=np.array([r['edited']['KL_change'] for f,r in rec]);predkl=np.array([r['edited']['KL_prediction'] for f,r in rec])
            ci=bootstrap_gain(control[valid],losses[valid],.0125) if valid.any() else [None,None]
            mse=np.mean([r['edited'][field+'_prediction_se'] for f,r in rec]);mse0=np.mean([r['edited'][field+'_nochange_se'] for f,r in rec])
            effects.append({'group':g['id'],'dose':dose,'metric':field,'n':len(rec),'valid_controls':int(valid.sum()),
                'mean_margin_loss':float(losses.mean()),'control_mean_margin_loss':float(control.mean()),'matched_excess_ci':ci,
                'top1_changes':sum(f['baseline']['top1']!=r['edited']['top1'] for f,r in rec),
                'mean_KL':float(basekl.mean()),'prediction_KL_ratio':float(predkl.mean()/max(basekl.mean(),1e-15)),
                'effect_prediction_mse_ratio':float(mse/max(mse0,1e-15)),
                'use_supported':bool(valid.mean()>=.95 and losses.mean()>=.2 and ci[0] is not None and ci[0]>0),
                'effect_prediction_supported':bool(mse<=.8*mse0),
                'failures':[{'family':f['sample']['family'],'prompt':f['sample']['prompt'],'baseline':f['baseline'],'record':r} for f,r in rec[:4]]})
    cap=read(OUT/'capability.json')['rows'];capstats={}
    for g in frozen['causal_groups'][:2]:
        orig=np.array([np.argmax(r['scores']['base'])==r['gold'] for r in cap]);edit=np.array([np.argmax(r['scores'][g['id']])==r['gold'] for r in cap]);capstats[g['id']]={'n':len(cap),'baseline_correct':int(orig.sum()),'lost':int((orig&~edit).sum()),'gained':int((~orig&edit).sum())}
    # Full-vocabulary prefix-only predictions, evaluated once on fresh families.
    kls=[];nulls=[]
    for f in range(NS['holdout']):
        scores=read(OUT/f'holdout-{f:04d}.json')['scores']
        kls.append(float(np.mean([s['distribution_KL'] for s in scores])));nulls.append(float(np.mean([s['distribution_baseline_KL'] for s in scores])))
        if (f+1)%64==0:log('output_prediction',f+1,NS['holdout'],start)
    doc={'schema':'native_puzzle_report.v1','kind':'real','run_id':RUN,'model_snapshot':SNAPSHOT,'provenance':p['provenance'],
         'plan_sha256':sha(OUT/'plan.json'),'frozen_sha256':sha(OUT/'frozen.json'),'groups':stats,'representatives':reps,'effects':effects,
         'behavior':read(OUT/'holdout-behavior.json'),'generation':read(OUT/'generation.json'),'capability':capstats,
         'output_prediction':{'KL':float(np.mean(kls)),'additive_KL':float(np.mean(nulls)),'per_family_KL':kls,'per_family_baseline':nulls},
         'restoration_checks':sum(r['restored_bitwise'] for r in causals),'limits':p['limits'],
         'scope':'controlled binding and output-language task;response explanation,causal use,andoutput prediction are separately assessed;no claim of universal semantic atoms'}
    save('report.json',doc);print(json.dumps({'groups_passed':sum(r['passed'] for r in stats),'groups':len(stats),'effects':[{k:r[k] for k in ['group','dose','mean_margin_loss','valid_controls','use_supported','effect_prediction_supported']} for r in effects],'output':{k:v for k,v in doc['output_prediction'].items() if not k.startswith('per_')}},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['freeze','pilot','discovery','select','calibrate','holdout','causal','capability','generation','report']);args=ap.parse_args()
    if args.stage=='freeze':freeze();return
    p=checked()
    if args.stage=='select':select(p);return
    if args.stage=='report':report(p);return
    torch.manual_seed(SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    with GpuLease(),torch.inference_mode():
        start=time.monotonic();log('load',0,1,start);m=load_model(p);log('load',1,1,start)
        if args.stage in NS:collect(m,tok,p,args.stage)
        else:globals()[args.stage](m,tok,p)
if __name__=='__main__':main()
