"""Real CUDA BF16 collection and prediction-before-intervention native replay."""
import argparse
import json
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import mlp_condition_design as D
from mlp_b34_confirm_run import bits,norm_match
from shared.dist_protocol.mechanism_trace import GpuLease

UNITS={29:17,30:2,33:1521,34:6097}

def log(stage,done,total,start):
    elapsed=monotonic()-start
    row={'stage':stage,'done':done,'total':total,'seconds':round(elapsed,2),
         'eta_seconds':round(elapsed*(total-done)/done,1) if done else None,
         'allocated_MiB':round(torch.cuda.memory_allocated()/2**20),'peak_MiB':round(torch.cuda.max_memory_allocated()/2**20)}
    try:
        cmd="(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | Where-Object { $_.InstanceName -like 'pid_"+str(os.getpid())+"_*' } | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
        p=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,timeout=10)
        row['shared_gpu_MiB']=round(float(p.stdout.strip())/2**20,2) if p.returncode==0 and p.stdout.strip() else None
    except Exception:row['shared_gpu_MiB']=None
    print(json.dumps(row),flush=True)
    with (D.OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n');f.flush()

class Assay:
    def __init__(self,model):
        self.model=model;self.layers=model.model.layers;self.mode=None;self.data={};self.args={};self.handles=[]
        for b in D.BLOCKS:
            m=self.layers[b]
            for f,module in [('G',m.mlp.gate_proj),('V',m.mlp.up_proj)]:
                self.handles.append(module.register_forward_hook(self.vector(b,f)))
            self.handles.append(m.mlp.down_proj.register_forward_pre_hook(self.full(b,'A')))
            self.handles.append(m.post_attention_layernorm.register_forward_pre_hook(self.full(b,'R')))
            self.handles.append(m.register_forward_hook(self.output(b)))
        for b in range(30,36):self.handles.append(self.layers[b].register_forward_pre_hook(self.arguments(b),with_kwargs=True))

    def vector(self,b,f):
        def h(module,args,output):
            if self.mode=='base':
                index=torch.arange(output.shape[0],device='cuda')[:,None]
                self.data[f'{f}{b}']=output[index,self.positions,UNITS[b]].clone()
        return h

    def full(self,b,f):
        def h(module,args):
            if self.mode in ('base','suffix'):self.data[f'{f}{b}']=args[0]
        return h

    def output(self,b):
        def h(module,args,value):
            if self.mode in ('base','suffix'):self.data[f'H{b}']=value
        return h

    def arguments(self,b):
        def h(module,args,kwargs):
            if self.mode=='base':self.args[b]=(args[1:],kwargs.copy())
        return h

    def observe(self,rows):
        self.mode='base';self.data={};self.args={}
        self.positions=torch.tensor([r['positions'] for r in rows],device='cuda')
        logits=self.model(input_ids=torch.tensor([r['tokens'] for r in rows],device='cuda'),use_cache=False,logits_to_keep=1).logits
        self.mode=None
        return logits,self.data.copy()

    def suffix(self,state,first):
        self.mode='suffix'
        try:
            for b in range(first,36):
                args,kwargs=self.args[b];state=self.layers[b](state,*args,**kwargs)
            # Preserve original shape at normalization, then keep final logits.
            return self.model.lm_head(self.model.model.norm(state)[:,-1:])
        finally:self.mode=None

    def source_edit(self,base,source,unit,delta):
        a=base[f'A{source}'].clone()
        old=float(a[0,-1,unit]);a[0,-1,unit]=(a[0,-1,unit].float()+delta).to(a.dtype)
        value=base[f'R{source}'][0,-1]+self.layers[source].mlp.down_proj(a)[0,-1]
        state=base[f'H{source}'].clone();state[0,-1]=value
        return state,float(a[0,-1,unit])-old

    def close(self):
        for h in self.handles:h.remove()

def save(name,body,plan):
    D.SAVE(D.OUT/name,{'provenance':plan['provenance'],'plan_sha256':D.HASH(D.OUT/'prereg.json'),**body})

def collect(model,plan,split):
    if split=='holdout':
        frozen=D.READ(D.OUT/'frozen-predictor.json')
        assert frozen['plan_sha256']==D.HASH(D.OUT/'prereg.json')
        assert frozen['calibration_sha256']==D.HASH(D.OUT/'discovery-calibration.npz')
    fs=D.families(plan,split);assay=Assay(model);start=monotonic();log(split,0,len(fs),start)
    try:
        for family in fs:
            number=family['family'];path=D.OUT/f'{split}-{number:04d}.npz';meta=path.with_suffix('.json')
            if meta.exists():
                doc=D.READ(meta);assert doc['sha256']==D.HASH(path) and doc['plan_sha256']==D.HASH(D.OUT/'prereg.json');continue
            arrays={f'A{b}':[] for b in D.BLOCKS};rows=[];selected=[]
            for offset in range(0,16,4):
                items=family['rows'][offset:offset+4];logits,base=assay.observe(items)
                ix=torch.arange(4,device='cuda')[:,None]
                selected.append(np.stack([np.stack([bits(base[f'G{b}']),bits(base[f'V{b}']),
                    bits(base[f'A{b}'][ix,assay.positions,UNITS[b]])],1) for b in D.BLOCKS],1))
                for b in D.BLOCKS:arrays[f'A{b}'].append(bits(base[f'A{b}'][:,-1]))
                score=logits[:,0,plan['choice_ids'][:2]].float().cpu().numpy()
                top=logits[:,0].argmax(-1).tolist()
                rows.extend({'id':r['id'],'scores':s.tolist(),'top1':t} for r,s,t in zip(items,score,top))
            np.savez_compressed(path,**{k:np.concatenate(v) for k,v in arrays.items()},selected=np.concatenate(selected))
            save(meta.name,{'sha256':D.HASH(path),'rows':rows,'prediction_frozen_sha256':D.HASH(D.OUT/'frozen-predictor.json') if split=='holdout' else None},plan)
            if (number+1)%16==0 or number+1==len(fs):log(split,number+1,len(fs),start)
            assert monotonic()-start<60*plan['resources']['gpu_stage_minutes'],'Stage budget exhausted'
    finally:assay.close()

def calibrate(model,plan):
    selection=D.READ(D.OUT/'selection.json');assay=Assay(model);start=monotonic()
    assert selection['behavior_gate_passed'],'Discovery behavior failed; preserve materials and do not open confirmation'
    fs=D.families(plan,'discovery')[:64];size=model.config.vocab_size
    sums=np.zeros((2,4,size),np.float64);den=np.zeros((2,4),np.float64);target=np.zeros((2,4),np.float64);checks=[]
    log('discovery_calibration',0,len(fs),start)
    try:
        for family in fs:
            row=family['rows'][(family['family']//4)%16]
            original,base=assay.observe([row]);stratum=row['role']*2+row['mapping']
            for p,(label,src,u,dst,v) in enumerate(D.PAIRS):
                assert torch.equal(assay.suffix(base[f'H{src}'],src+1),original),('zero_replay',row['id'],src)
                for sign in (-1,1):
                    state,delta=assay.source_edit(base,src,u,sign*.5*selection[label]['scale'])
                    logits=assay.suffix(state,src+1)
                    change=(logits-original).float()[0,0].cpu().numpy()
                    # Center full-vocabulary logits: irrelevant constant shifts have no effect.
                    change-=change.mean()
                    sums[p,stratum]+=delta*change;den[p,stratum]+=delta**2
                    target[p,stratum]+=delta*float(assay.data[f'A{dst}'][0,-1,v].float()-base[f'A{dst}'][0,-1,v].float())
                assert torch.equal(assay.suffix(base[f'H{src}'],src+1),original)
            checks.append(row['id'])
            if len(checks)%16==0:log('discovery_calibration',len(checks),len(fs),start)
        assert (den>0).all(),den
        np.savez(D.OUT/'discovery-calibration.npz',logit_slope=(sums/den[:,:,None]).astype(np.float32),target_slope=target/den)
        save('frozen-predictor.json',{'selection_sha256':D.HASH(D.OUT/'selection.json'),
             'calibration_sha256':D.HASH(D.OUT/'discovery-calibration.npz'),'checks':checks,
             'frozen_before_holdout':not list(D.OUT.glob('holdout-[0-9]*.npz'))},plan)
    finally:assay.close()

def distribution(logits,base,pred,ids,gold):
    q=logits.float()[0,0].log_softmax(-1);p=base.float()[0,0].log_softmax(-1);r=pred.log_softmax(-1)
    return {'scores':logits[0,0,ids[:2]].float().tolist(),'top1':int(q.argmax()),
       'KL_base_to_edit':float((p.exp()*(p-q)).sum()),'KL_edit_to_prediction':float((q.exp()*(q-r)).sum()),
       'KL_edit_to_nochange':float((q.exp()*(q-p)).sum()),'gold_nll_change':float(p[ids[gold]]-q[ids[gold]])}

def intervene(model,plan):
    selection=D.READ(D.OUT/'selection.json');frozen=D.READ(D.OUT/'frozen-predictor.json')
    assert frozen['selection_sha256']==D.HASH(D.OUT/'selection.json') and frozen['calibration_sha256']==D.HASH(D.OUT/'discovery-calibration.npz')
    co=np.load(D.OUT/'discovery-calibration.npz');assay=Assay(model);fs=D.families(plan,'holdout');start=monotonic()
    log('intervention',0,len(fs),start)
    try:
        for family in fs:
            f=family['family'];path=D.OUT/f'causal-{f:04d}.json'
            if path.exists():
                assert D.READ(path)['predictor_sha256']==D.HASH(D.OUT/'frozen-predictor.json');continue
            row=family['rows'][(f//4)%16];original,base=assay.observe([row]);stratum=row['role']*2+row['mapping']
            base_scores=original[0,0,plan['choice_ids'][:2]].float().tolist();winner=int(np.argmax(base_scores));other=1-winner
            predictions=[];curves=[]
            for p,(label,src,u,dst,v) in enumerate(D.PAIRS):
                slope=torch.tensor(co['logit_slope'][p,stratum],device='cuda')
                sign=1 if float(slope[plan['choice_ids'][other]]-slope[plan['choice_ids'][winner]])>=0 else -1
                states=[]
                for dose in D.DOSES:
                    state,delta=assay.source_edit(base,src,u,sign*dose*selection[label]['scale'])
                    states.append((dose,state,delta))
                    predictions.append({'pair':label,'dose':dose,'native_delta':delta,
                        'target_delta':float(delta*co['target_slope'][p,stratum]),
                        'desired_margin_delta':float(delta*(slope[plan['choice_ids'][other]]-slope[plan['choice_ids'][winner]])),
                        'predicted_top1':int((original.float()[0,0]+delta*slope).argmax())})
                # Durable commitment before running the edited suffix, per candidate pair.
                save(f'prediction-{f:04d}-{label}.json',{'sample_id':row['id'],'predictor_sha256':D.HASH(D.OUT/'frozen-predictor.json'),
                    'rows':[x for x in predictions if x['pair']==label],'available_inputs':'frozen discovery calibration, baseline state/logits, planned native delta'},plan)
                assert torch.equal(assay.suffix(base[f'H{src}'],src+1),original),('zero_replay',row['id'],label)
                rng=torch.Generator(device='cuda').manual_seed(D.SEED+f*2+p)
                for dose,state,delta in states:
                    control,norm,cnorm,valid=norm_match(base[f'H{src}'][0,-1],state[0,-1],rng)
                    rec={'pair':label,'dose':dose,'norm':norm,'control_norm':cnorm,'norm_valid':valid,
                         'prediction':next(x for x in predictions if x['pair']==label and x['dose']==dose)}
                    edited=assay.suffix(state,src+1)
                    rec['edit']=distribution(edited,original,original.float()[0,0]+delta*slope,plan['choice_ids'],row['gold'])
                    rec['target_delta']=float(assay.data[f'A{dst}'][0,-1,v].float()-base[f'A{dst}'][0,-1,v].float())
                    kept={k:assay.data[k] for k in (f'A{dst}',f'R{dst}',f'H{dst}')}
                    null=base[f'H{src}'].clone();null[0,-1]=control
                    null_logits=assay.suffix(null,src+1)
                    rec['control']=distribution(null_logits,original,original.float()[0,0],plan['choice_ids'],row['gold'])
                    rec['control_target_delta']=float(assay.data[f'A{dst}'][0,-1,v].float()-base[f'A{dst}'][0,-1,v].float())
                    if dose==D.PRIMARY:
                        restored=kept[f'A{dst}'].clone();restored[0,-1,v]=base[f'A{dst}'][0,-1,v]
                        value=kept[f'R{dst}'][0,-1]+assay.layers[dst].mlp.down_proj(restored)[0,-1]
                        sham,n,cn,ok=norm_match(kept[f'H{dst}'][0,-1],value,rng)
                        rec['rescue_norm_valid']=ok;rec['rescue_norm']=n;rec['sham_norm']=cn
                        for key,val in [('rescue',value),('sham',sham)]:
                            h=kept[f'H{dst}'].clone();h[0,-1]=val
                            lo=assay.suffix(h,dst+1)
                            rec[key]={'scores':lo[0,0,plan['choice_ids'][:2]].float().tolist(),'top1':int(lo[0,0].argmax())}
                    curves.append(rec)
                assert torch.equal(assay.suffix(base[f'H{src}'],src+1),original)
            save(path.name,{'sample_id':row['id'],'family':f,'gold':row['gold'],'base_scores':base_scores,
               'base_top1':int(original[0,0].argmax()),'predictor_sha256':D.HASH(D.OUT/'frozen-predictor.json'),
               'baseline_correct':winner==row['gold'],'low_margin':abs(base_scores[0]-base_scores[1])<=2.,
               'restore_bitwise':True,'curves':curves,
               'prediction_hashes':{label:D.HASH(D.OUT/f'prediction-{f:04d}-{label}.json') for label,*_ in D.PAIRS}},plan)
            if (f+1)%16==0:log('intervention',f+1,len(fs),start)
            assert monotonic()-start<60*plan['resources']['gpu_stage_minutes']
    finally:assay.close()

def capability(model,plan):
    assert D.HASH(D.CAP)==plan['capability']['sha256']
    selection=D.READ(D.OUT/'selection.json');tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True);tok.padding_side='left'
    items=[]
    for row in D.READ(D.CAP):
        if not row['gate_eligible']:continue
        text=row['question']+'\n'+'\n'.join(f'{k}. {v}' for k,v in zip('ABCDE',row['choices']))+'\n请选择一个正确选项字母。'
        prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
        items.append({'id':row['id'],'source':row['source'],'gold':'ABCDE'.index(row['answer']),'n':len(row['choices']),'tokens':tok.encode(prompt,add_special_tokens=False)})
    save('capability-materials.json',{'rows':items},plan);rows=[];start=monotonic();log('capability',0,len(items),start)
    for offset in range(0,len(items),8):
        chunk=items[offset:offset+8];batch=tok.pad({'input_ids':[x['tokens'] for x in chunk]},return_tensors='pt').to('cuda');scores={}
        for label,src,u,sign in [('base',29,17,0),*[(p,b,u,s) for p,b,u,_,_ in D.PAIRS for s in (-1,1)]]:
            def patch(module,args):
                x=args[0].clone();x[:,-1,u]=(x[:,-1,u].float()+sign*selection[label]['scale']).to(x.dtype);return (x,)
            handle=model.model.layers[src].mlp.down_proj.register_forward_pre_hook(patch) if sign else None
            try:scores[label+str(sign)]=model(**batch,use_cache=False,logits_to_keep=1).logits[:,0,plan['choice_ids']].float().cpu().numpy()
            finally:
                if handle:handle.remove()
        rows.extend({'id':r['id'],'source':r['source'],'gold':r['gold'],'scores':{k:v[i,:r['n']].tolist() for k,v in scores.items()}} for i,r in enumerate(chunk))
        if len(rows)%128==0 or len(rows)==len(items):log('capability',len(rows),len(items),start)
    save('capability.json',{'selection_sha256':D.HASH(D.OUT/'selection.json'),'rows':rows},plan)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['discovery','calibrate','holdout','intervene','capability']);stage=parser.parse_args().stage
    plan=D.checked();torch.manual_seed(D.SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with GpuLease(),torch.inference_mode():
        start=monotonic();log('load',0,1,start)
        model=AutoModelForCausalLM.from_pretrained(D.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
        log('load',1,1,start)
        if stage in ('discovery','holdout'):collect(model,plan,stage)
        else:{'calibrate':calibrate,'intervene':intervene,'capability':capability}[stage](model,plan)

if __name__=='__main__':main()
