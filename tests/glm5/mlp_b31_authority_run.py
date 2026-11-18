"""Collect and intervene on a frozen Qwen3-4B BF16 authorization task."""
import argparse
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from time import monotonic

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mlp_b31_authority_design as D
from mlp_b34_confirm_run import bits, choices, log, norm_match
from shared.dist_protocol.mechanism_trace import GpuLease

OUT, SAVE, READ, HASH = D.OUT, D.SAVE, D.READ, D.HASH


class Assay:
    def __init__(self,model,ids):
        self.model=model
        self.decoder=model.model
        self.layers=self.decoder.layers
        self.ids=ids
        self.data={}
        self.mode=None
        self.args={}
        self.handles=[]
        for block in (31,32):
            module=self.layers[block]
            self.handles.extend([
                module.post_attention_layernorm.register_forward_pre_hook(self.tap(f'R{block}',block,True)),
                module.mlp.down_proj.register_forward_pre_hook(self.tap(f'A{block}',block,True)),
                module.register_forward_hook(self.tap(f'H{block}',block)),
            ])
        for block in range(32,36):
            self.handles.append(self.layers[block].register_forward_pre_hook(self.arguments(block),with_kwargs=True))

    def arguments(self,block):
        def capture(module,args,kwargs):
            if self.mode=='base':
                self.args[block]=(args[1:],kwargs.copy())
        return capture

    def tap(self,name,block,input=False):
        def capture(module,args,value=None):
            if self.mode=='base' or (self.mode=='suffix' and block==32):
                self.data[name]=args[0] if input else value
        return capture

    def observe(self,row):
        self.data={}
        self.args={}
        self.mode='base'
        tokens=torch.tensor(row['tokens'],device='cuda')
        logits=self.model(input_ids=tokens,use_cache=False,logits_to_keep=1).logits
        self.mode=None
        return logits,self.data.copy()

    def suffix(self,state,first=32):
        self.mode='suffix'
        try:
            for block in range(first,36):
                args,kwargs=self.args[block]
                state=self.layers[block](state,*args,**kwargs)
            return self.model.lm_head(self.decoder.norm(state)[:,-1:])
        finally:
            self.mode=None

    def close(self):
        for handle in self.handles:
            handle.remove()

    def intervene(self,row,baseline,original,selection,seed):
        source,target=self.layers[31].mlp.down_proj,self.layers[32].mlp.down_proj
        h31=baseline['H31']
        a31,a32=baseline['A31'],baseline['A32']
        assert torch.equal(baseline['R31'][:,-1:] + source(a31)[:,-1:], h31[:,-1:]),row['id']
        noop=self.suffix(h31)
        assert torch.equal(noop,original) and torch.equal(self.data['A32'][:,-1],a32[:,-1]), ('zero_replay',row['id'],float((noop.float()-original.float()).abs().max()))
        recipient=row['recipient_index']
        units=selection['B31']['units']
        target_units=selection['B32']['units']
        shift=torch.tensor(selection['B31']['write'],device='cuda')*(1-2*recipient)
        rng=torch.Generator(device='cuda').manual_seed(seed)
        result=[]
        for dose in D.DOSES:
            patched=a31.clone()
            patched[recipient,-1,units]=(patched[recipient,-1,units].float()+dose*shift).to(patched.dtype)
            edited=baseline['R31'][recipient,-1]+source(patched)[recipient,-1]
            null, magnitude, null_magnitude, valid=norm_match(h31[recipient,-1],edited,rng)
            record={'dose':dose,'norm':magnitude,'control_norm':null_magnitude,'norm_valid':valid}
            kept={}
            for name,value in [('edit',edited),('control',null)]:
                state=h31.clone()
                state[recipient,-1]=value
                logits=self.suffix(state)
                target_delta=self.data['A32'][recipient,-1].float()-a32[recipient,-1].float()
                record[name]={'scores':choices(logits,self.ids[:2])[recipient].tolist(),
                              'top1':int(logits[recipient,0].argmax()),
                              'target_delta':target_delta[target_units].tolist(),
                              'all_target_delta_norm':float(target_delta.norm())}
                if name=='edit' and dose==D.PRIMARY:
                    kept={k:self.data[k] for k in ('A32','R32','H32')}
            if kept:
                restored=kept['A32'].clone()
                restored[recipient,-1,target_units]=a32[recipient,-1,target_units]
                assert torch.equal(kept['R32'][:,-1:]+target(kept['A32'])[:,-1:],kept['H32'][:,-1:]),row['id']
                restored_value=kept['R32'][recipient,-1]+target(restored)[recipient,-1]
                sham_value,norm,sham_norm,valid_sham=norm_match(kept['H32'][recipient,-1],restored_value,rng)
                record['rescue']={'norm':norm,'sham_norm':sham_norm,'norm_valid':valid_sham}
                for label,value in [('scores',restored_value),('sham_scores',sham_value)]:
                    state=kept['H32'].clone()
                    state[recipient,-1]=value
                    logits=self.suffix(state,33)
                    record['rescue'][label]=choices(logits,self.ids[:2])[recipient].tolist()
            result.append(record)
        assert torch.equal(self.suffix(h31),original) and torch.equal(self.data['A32'][:,-1],a32[:,-1]),('restore',row['id'])
        return {'id':row['id'],'base_scores':choices(original,self.ids[:2]).tolist(),
                'base_top1':original[:,0].argmax(1).tolist(),'restore_bitwise':True,'doses':result}


def identity():
    return HASH(OUT/'prereg.json')


def selection(plan):
    data=READ(OUT/'selection.json')
    assert data['baseline_gate'] and data['plan_sha256']==identity() and data['provenance']==plan['provenance']
    for record in data['discovery_chunks']:
        assert HASH(OUT/record['path'])==record['sha256']
    return data


def pairs(model,plan,split,selected=None):
    if split=='holdout':
        replay=READ(OUT/'calibration.json')
        assert replay['plan_sha256']==identity() and replay['selection_sha256']==HASH(OUT/'selection.json') and replay['passed']
    items=D.items(plan,split)
    assay=Assay(model,plan['choice_ids'])
    started=monotonic()
    log(split,0,len(items),started)
    try:
        if split=='calibration':
            raise ValueError('Use calibrate')
        for start in range(0,len(items),16):
            raw=OUT/f'{split}-{start:04d}.npz'
            meta=raw.with_suffix('.json')
            selected_sha=HASH(OUT/'selection.json') if selected else None
            if meta.exists():
                doc=READ(meta)
                assert doc['raw_sha256']==HASH(raw) and doc['selection_sha256']==selected_sha and doc['plan_sha256']==identity()
                continue
            rows,arrays=[],{'A31':[],'A32':[]}
            for index,row in enumerate(items[start:start+16],start):
                logits,base=assay.observe(row)
                for key in arrays:
                    arrays[key].append(bits(base[key][:,-1]))
                rows.append(assay.intervene(row,base,logits,selected,D.SEED+index) if selected else
                    {'id':row['id'],'base_scores':choices(logits,plan['choice_ids'][:2]).tolist(),
                     'base_top1':logits[:,0].argmax(1).tolist()})
                if (index+1)%8==0:
                    log(split,index+1,len(items),started)
                if monotonic()-started>60*plan['resources']['stage_minutes']:
                    raise RuntimeError('GPU stage time budget exceeded')
            np.savez(raw,**{k:np.stack(v) for k,v in arrays.items()})
            SAVE(meta,{'provenance':plan['provenance'],'plan_sha256':identity(),'selection_sha256':selected_sha,
                       'raw_sha256':HASH(raw),'rows':rows})
    finally:
        assay.close()


def calibrate(model,plan,selected):
    assay=Assay(model,plan['choice_ids'])
    started=monotonic()
    log('calibration',0,12,started)
    checks=[]
    try:
        for i,row in enumerate(D.items(plan,'discovery')[:12]):
            logits,base=assay.observe(row)
            result=assay.intervene(row,base,logits,selected,D.SEED+i)
            checks.append({'id':row['id'],'restored':result['restore_bitwise'],
                           'valid_doses':sum(d['norm_valid'] for d in result['doses'])})
            log('calibration',i+1,12,started)
    finally:
        assay.close()
    SAVE(OUT/'calibration.json',{'provenance':plan['provenance'],'plan_sha256':identity(),
        'selection_sha256':HASH(OUT/'selection.json'),'passed':True,'checks':checks})


def capability(model,plan,selected):
    if (OUT/'capability.json').exists():
        raise RuntimeError('Frozen capability exists')
    assert HASH(D.CAP)==plan['capability']['sha256']
    tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True)
    tok.padding_side='left'
    items=[]
    for row in READ(D.CAP):
        if not row['gate_eligible']:
            continue
        options='\n'.join(f'{k}. {v}' for k,v in zip('ABCDE',row['choices']))
        text=row['question']+'\n'+options+'\n请选择一个正确选项字母。'
        prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
        items.append({'id':row['id'],'source':row['source'],'tokens':tok.encode(prompt,add_special_tokens=False),
                      'correct':'ABCDE'.index(row['answer']),'choices':len(row['choices'])})
    assert len(items)==plan['capability']['n']
    SAVE(OUT/'capability-materials.json',{'provenance':plan['provenance'],'rows':items})
    module=model.model.layers[31].mlp.down_proj
    units=selected['B31']['units']
    vector=torch.tensor(selected['B31']['write'],device='cuda')*D.PRIMARY
    started=monotonic()
    log('capability',0,len(items),started)
    rows=[]
    for start in range(0,len(items),8):
        chunk=items[start:start+8]
        batch=tok.pad({'input_ids':[r['tokens'] for r in chunk]},return_tensors='pt').to('cuda')
        scores={}
        for tag,sign in [('base',0),('positive',1),('negative',-1)]:
            def patch(module,args):
                x=args[0].clone()
                x[:,-1,units]=(x[:,-1,units].float()+sign*vector).to(x.dtype)
                return (x,)
            handle=module.register_forward_pre_hook(patch) if sign else None
            try:
                scores[tag]=choices(model(**batch,use_cache=False,logits_to_keep=1).logits,plan['choice_ids']).tolist()
            finally:
                if handle:
                    handle.remove()
        rows.extend({'id':r['id'],'source':r['source'],'gold':r['correct'],
                     'scores':{k:v[i][:r['choices']] for k,v in scores.items()}} for i,r in enumerate(chunk))
        if len(rows)%64==0 or len(rows)==len(items):
            log('capability',len(rows),len(items),started)
        if monotonic()-started>60*plan['resources']['stage_minutes']:
            raise RuntimeError('GPU capability time budget exceeded')
    SAVE(OUT/'capability.json',{'provenance':plan['provenance'],'plan_sha256':identity(),
         'selection_sha256':HASH(OUT/'selection.json'),'materials_sha256':HASH(OUT/'capability-materials.json'),'rows':rows})


def run(stage):
    plan=D.checked()
    selected=None if stage=='discovery' else selection(plan)
    torch.manual_seed(D.SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started=monotonic()
    with GpuLease(),torch.inference_mode():
        log('load',0,1,started,requested=stage)
        model=AutoModelForCausalLM.from_pretrained(D.MODEL,local_files_only=True,dtype=torch.bfloat16,
            device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
        log('load',1,1,started,requested=stage)
        if stage=='capability':
            capability(model,plan,selected)
        elif stage=='calibrate':
            calibrate(model,plan,selected)
        else:
            pairs(model,plan,stage,selected)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['discovery','calibrate','holdout','capability'])
    run(parser.parse_args().stage)
