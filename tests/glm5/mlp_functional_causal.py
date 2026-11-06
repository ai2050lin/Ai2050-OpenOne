"""Actual upstream write ablation, natural generation and paired capability checks."""
import argparse
from collections import Counter
import json
import time

import numpy as np
import torch
from scipy.stats import binomtest

import mlp_functional_common as C
from mlp_fidelity_v2_collect_v4 import first_word, greedy
from mlp_fidelity_v2_tail import capability_interval, cluster_interval
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease
from shared.dist_protocol.natural_register_runner import snapshot
from shared.dist_protocol.transcoder import TopKTranscoder


def edited(model, rows, positions, heads, mlps, alpha, read_prequestion, dictionary=None):
    handles=[];r=[]
    try:
        if read_prequestion:
            def read(_,args):r.append(args[0][rows,positions].detach().float())
            handles.append(model.model.layers[35].post_attention_layernorm.register_forward_pre_hook(read))
        for b,h in heads:
            def drop_head(_,args,h=h):
                x=args[0].clone();sl=slice(h*128,(h+1)*128)
                x[rows,positions,sl]=(x[rows,positions,sl].float()*(1-alpha)).to(x.dtype)
                return (x,*args[1:])
            handles.append(model.model.layers[b].self_attn.o_proj.register_forward_pre_hook(drop_head))
        for b in mlps:
            def drop_mlp(_,args,y):
                out=y.clone();out[rows,positions]=(out[rows,positions].float()*(1-alpha)).to(y.dtype)
                return out
            handles.append(model.model.layers[b].mlp.register_forward_hook(drop_mlp))
        if dictionary is not None:
            def replace_mlp(_,args,y):
                out=y.clone();out[rows,positions]=dictionary(y[rows,positions].float()).to(y.dtype)
                return out
            handles.append(model.model.layers[35].mlp.register_forward_hook(replace_mlp))
        yield r
    finally:
        for handle in handles:handle.remove()


from contextlib import contextmanager
edited=contextmanager(edited)


def summarize_controlled(rows, eligible):
    summary={}
    for q,name in [(0,'location'),(1,'industry')]:
        qs=[r for r in rows if r['q']==q];names=list(qs[0]['conditions']);base=np.array([r['conditions']['clean']['correct'] for r in qs],bool)
        per={}
        for op in names:
            vals=[r['conditions'][op] for r in qs];pred=np.array([v['correct'] for v in vals],bool)
            loses=int((base&~pred).sum());gains=int((~base&pred).sum());losses=base.astype(float)-pred
            per[op]={'accuracy':float(pred.mean()),'loss':cluster_interval(losses,C.SEED),'loses':loses,'gains':gains,
                     'mcnemar_exact_p':binomtest(loses,loses+gains,.5).pvalue if loses+gains else 1.,
                     **{f'mean_{k}':float(np.mean([v[k] for v in vals])) for k in ['KL','answer_gap','signed_location_residual','signed_industry_residual']}}
        control=per.get('random_1');joint=per['joint_1']
        if control:
            spec=np.array([int(r['conditions']['random_1']['correct'])-int(r['conditions']['joint_1']['correct']) for r in qs])
            specificity=cluster_interval(spec,C.SEED+7)
        else:specificity=None
        summary[name]={'operations':per,'target_minus_random':specificity,'n':len(qs),
                       'passed':bool(q==0 and eligible and specificity and joint['loss']['mean']>=.10 and joint['mcnemar_exact_p']<.05 and specificity['mean']>=.05)}
    return summary


def run(part):
    torch.set_num_threads(4);start=time.time()
    result=C.OUT/f'{part}-causal.json'
    if result.exists():return
    path=C.OUT/'path-analysis.json';analysis=C.read(path)
    source=C.OUT/'controlled-inputs.json';corpus=C.read(source)['rows'];train=C.read(C.OUT/'dictionary-training.json')
    primary=analysis['selected']['location'];heads=[(p['block'],p['head']) for p in primary if p['head'] is not None]
    mlps=[p['block'] for p in primary if p['head'] is None]
    rnd_head=[(p['block'],p['head']) for p in analysis['random'] if p['head'] is not None]
    rnd_mlp=[p['block'] for p in analysis['random'] if p['head'] is None]
    offsets=Counter(r['assistant']-r['prequestion'] for r in corpus if r['split']=='discovery')
    offset=offsets.most_common(1)[0][0]
    if part=='generated':
        take={b:int(C.digest([C.SEED,b,'causal-case'])[:8],16)%4 for b in range(288,448)}
        items=[r for r in corpus if r['b'] in take and r['r']*2+r['t']==take[r['b']]]
        assert len(items)==320
        ops=[('clean',[],[],0.),('joint_0.5',heads,mlps,.5),('joint_1',heads,mlps,1.),('attention_1',heads,[],1.),('mlp_1',[],mlps,1.)]
        if analysis['random_matched']:ops.append(('random_1',rnd_head,rnd_mlp,1.))
    else:
        items=C.read(C.CAP);assert sum(r['gate_eligible'] for r in items)==2200
        ops=[('clean',[],[],0.),('functional_sae',[],[],0.),('upstream_joint',heads,mlps,1.)]
    eligible=analysis['behavior_gate'] and bool(heads) and bool(mlps)
    contract={**C.identity(),'schema':'functional_upstream_causal_prereg.v1','part':part,'source_sha256':C.file_sha(__file__),
              'paths_sha256':C.file_sha(path),'controlled_sha256':C.file_sha(source),'capability_input_sha256':C.file_sha(C.CAP),
              'dictionary_sha256':train['outputs']['functional_sae']['sha256'],'operations':ops,'n':len(items),
              'upstream_eligible':eligible,'random_matched':analysis['random_matched'],'capability_offset_from_last_token':offset,
              'scope':'prequestion single token writers with full recomputation of every downstream layer and KV; capacity matches a frozen discovery offset, not a semantic boundary.'}
    C.save(C.OUT/f'{part}-causal-prereg.json',contract)
    if not eligible:
        if part=='generated':
            C.save(result,{**contract,'status':'not_eligible','summary':{},'rows':[]});return
        ops=ops[:2]
    output=[];canaries=[]
    # Resume only at fully committed batches, and never change their design.
    progress=C.OUT/f'{part}-causal-progress.jsonl'
    if progress.exists():output=[json.loads(line) for line in progress.read_text(encoding='utf-8').splitlines()]
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        d=torch.tensor([analysis['directions'][a]['direction'] for a in ['location','industry']],device='cuda')
        cutoff=torch.tensor([analysis['directions'][a]['threshold'] for a in ['location','industry']],device='cuda')
        dictionary=None
        if part=='capability':
            p=C.OUT/train['outputs']['functional_sae']['path'];assert C.file_sha(p)==contract['dictionary_sha256']
            with np.load(p) as z:dictionary=TopKTranscoder.from_state(z,device='cuda')
        choices={w:sorted({tok.encode(x,add_special_tokens=False)[0] for x in [w,w.capitalize(),' '+w,' '+w.capitalize()]}) for w in ['coastal','inland','fishing','mining']}
        letter_ids=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABCDE']
        bs=4 if part=='generated' else 12
        for lo in range(len(output),len(items),bs):
            batch=items[lo:lo+bs];length=len(batch);indices=torch.arange(length,device='cuda')
            if part=='generated':
                inputs={'input_ids':torch.tensor([r['tokens'] for r in batch],device='cuda')}
                pos=torch.tensor([r['prequestion'] for r in batch],device='cuda')
            else:
                prompts=[]
                for r in batch:
                    text=r['question']+'\n'+'\n'.join(f'{a}. {b}' for a,b in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。'
                    prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                          add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
                inputs={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()}
                length_full=inputs['input_ids'].shape[1]
                pos=torch.tensor([max(length_full-len(x),length_full-1-offset) for x in prompts],device='cuda')
            native_logits=None;per={}
            for name,head_group,mlp_group,alpha in ops:
                replace=dictionary if name=='functional_sae' else None
                positions=torch.full_like(pos,inputs['input_ids'].shape[1]-1) if replace is not None else pos
                with edited(model,indices,positions,head_group,mlp_group,alpha,part=='generated',replace) as read:
                    o=model(**inputs,use_cache=part=='generated',logits_to_keep=1)
                logits=o.logits[:,-1].float()
                if lo==0 and name=='clean':
                    with edited(model,indices,pos,heads,mlps,0.,False):
                        zero=model(**inputs,use_cache=part=='generated',logits_to_keep=1).logits[:, -1].float()
                    assert torch.equal(zero,logits),'zero dose changed original logits'
                    canaries.append({'zero_dose_actual_hook_logit_equal':True})
                if part=='generated':
                    assert len(read)==1
                    lp=logits.log_softmax(-1)
                    if native_logits is None:native_logits=lp
                    kl=(native_logits.exp()*(native_logits-lp)).sum(-1).cpu().tolist()
                    margins=(read[0]@d.T-cutoff).cpu().numpy()
                    tokens=greedy(model,o.past_key_values,logits,eos,limit=4)
                    per[name]=[]
                    for j,(item,ids) in enumerate(zip(batch,tokens)):
                        text=tok.decode(ids,skip_special_tokens=True);word=first_word(text)
                        gap=float(lp[j,choices[item['expected']]].max()-lp[j,choices[item['contrast']]].max())
                        per[name].append({'generated':text,'word':word,'generated_ids':ids,'correct':word==item['expected'],'KL':kl[j],'answer_gap':gap,
                            'signed_location_residual':float(margins[j,0]*(2*item['t']-1)),'signed_industry_residual':float(margins[j,1]*(2*item['i']-1))})
                else:
                    per[name]=[]
                    for j,item in enumerate(batch):
                        pred='ABCDE'[int(logits[j,letter_ids[:len(item['choices'])]].argmax())]
                        per[name].append({'prediction':pred,'correct':pred==item['answer'],'intervention_position':int(positions[j]),
                                          'actual_offset':int(inputs['input_ids'].shape[1]-1-positions[j])})
                del o,logits
            records=[]
            for j,item in enumerate(batch):
                record={k:v for k,v in item.items() if k!='tokens'} if part=='generated' else {k:item.get(k) for k in ['id','source','answer','gate_eligible']}
                records.append({**record,'index':lo+j,'conditions':{name:values[j] for name,values in per.items()}})
            output.extend(records)
            with progress.open('a',encoding='utf-8') as f:
                f.writelines(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records)
            if (lo+length)%(32 if part=='generated' else 120)==0:
                print(part,lo+length,'/',len(items),'seconds',round(time.time()-start,1),flush=True)
        assert snapshot(C.MODEL)==contract['model_snapshot']
    if part=='generated':summary=summarize_controlled(output,eligible)
    else:
        mask=np.asarray([r['gate_eligible'] for r in output],bool)
        base=[r['conditions']['clean']['correct'] for r in output]
        summary={name:capability_interval(base,[r['conditions'][name]['correct'] for r in output],mask)
                 for name,*_ in ops if name!='clean'}
        summary['reference_only_n']=int((~mask).sum())
    C.save(result,{**contract,'status':'complete','summary':summary,'rows':output,'canaries':canaries,'seconds':time.time()-start})
    print(part,summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['generated','capability']);run(p.parse_args().part)
