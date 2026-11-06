"""Intervene on native writes, then replay and rescue the exact autoregressive cache slots."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import time

import numpy as np
import torch
from scipy.stats import binomtest

import mlp_kv_common as C
from mlp_kv_behavior import frozen, score, word_indices
from mlp_kv_materials import CUT
from mlp_fidelity_v2_collect_v4 import greedy
from mlp_fidelity_v2_tail import capability_interval, cluster_interval
from mlp_functional_causal import edited
from mlp_functional_trace import bit_array, load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def prefix(model,x,heads=(),mlps=(),alpha=1.):
    rows=torch.arange(len(x),device=x.device);positions=torch.full_like(rows,CUT-1);fields={}
    def capture(_,args):fields['incoming_b35']=args[0][:,-1].detach().clone()
    def head(_,args):fields['head34_2']=args[0][:,-1,2*128:3*128].detach().clone()
    # This observer sits before the edit; the actual KV is read after the full prefix forward.
    hooks=[model.model.layers[35].input_layernorm.register_forward_pre_hook(capture),
           model.model.layers[34].self_attn.o_proj.register_forward_pre_hook(head)]
    try:
        with edited(model,rows,positions,heads,mlps,alpha,True) as local:
            output=model(input_ids=x[:,:CUT],use_cache=True,logits_to_keep=1)
    finally:
        for hook in hooks:hook.remove()
    fields['pre_mlp_b35']=local[0]
    return output.past_key_values,fields


def slot(cache,b,k):
    layer=cache.layers[b]
    return (layer.keys if k=='K' else layer.values)[:,:,CUT-1,:]


def cache_changes(clean,edit,allowed):
    changed=[];other_positions=True
    for b,(a,z) in enumerate(zip(clean.layers,edit.layers)):
        for kind,v,w in [('K',a.keys,z.keys),('V',a.values,z.values)]:
            if torch.equal(v,w):continue
            changed.append((b,kind))
            other_positions &= torch.equal(v[:,:,:CUT-1],w[:,:,:CUT-1])
    return {'slots':changed,'only_allowed_layers':all(b in allowed for b,_ in changed),
            'all_earlier_positions_bitwise_equal':other_positions}


@contextmanager
def patched(cache,overrides):
    assert cache.get_seq_length()==CUT
    backup={key:slot(cache,*key).clone() for key in overrides}
    try:
        for key,value in overrides.items():slot(cache,*key).copy_(value)
        yield
    finally:
        cache.crop(CUT)
        for key,value in backup.items():slot(cache,*key).copy_(value)


def branch(model,x,cache,eos,overrides=None,generate=True):
    with patched(cache,overrides or {}):
        out=model(input_ids=x[:,CUT:],past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits=out.logits[:,-1].detach().float()
        tokens=greedy(model,out.past_key_values,logits,eos,limit=4) if generate else None
    return logits,tokens


def take(cache,layers):return {(b,k):slot(cache,b,k).clone() for b in layers for k in ['K','V']}


def control(clean,changed):
    out={}
    for (b,k),value in changed.items():
        base=clean[b,k];delta=value.float()-base.float();rng=np.random.default_rng(C.SEED+b*17+(k=='V'))
        perm=torch.as_tensor(rng.permutation(delta.shape[-1]),device=base.device)
        signs=torch.as_tensor(rng.choice(np.asarray([-1,1],np.float32),size=delta.shape[-1]),device=base.device)
        out[b,k]=(base.float()+delta[...,perm]*signs).to(base.dtype)
    return out


def equal_rows(a,b):return (a==b).all(-1).cpu().tolist()


def metrics(rows,conditions,tok,word_ids):
    clean=conditions['clean'][0].double().log_softmax(-1);answer=[]
    for j,r in enumerate(rows):answer.append({k:v for k,v in r.items() if k!='tokens'}|{'conditions':{}})
    for name,(logits,generated) in conditions.items():
        q=logits.double().log_softmax(-1);kl=(clean.exp()*(clean-q)).sum(-1).cpu().tolist()
        for j,r in enumerate(rows):answer[j]['conditions'][name]=score(r,logits[j],generated[j],tok,word_ids)|{'KL':float(max(0.,kl[j]))}
    return answer


def summarize(rows,gate,minimum):
    groups={'all':rows,'confirm':[r for r in rows if r['split']=='confirm'],'lockbox':[r for r in rows if r['split']=='lockbox'],
            'low_margin':[r for r in rows if r['conditions']['clean']['correct'] and 0<r['conditions']['clean']['gap']<=8]}
    report={}
    for name,group in groups.items():
        if not group:
            report[name]={'n':0,'eligible':False};continue
        base=np.asarray([r['conditions']['clean']['correct'] for r in group],bool)
        ops={}
        for op in group[0]['conditions']:
            v=[r['conditions'][op] for r in group];actual=np.asarray([x['correct'] for x in v],bool)
            losses=int((base&~actual).sum());gains=int((~base&actual).sum())
            ops[op]={'accuracy':float(actual.mean()),'losses':losses,'gains':gains,
                     'net_loss':float(base.mean()-actual.mean()),'mcnemar_exact_p':float(binomtest(losses,losses+gains).pvalue) if losses+gains else 1.,
                     'mean_KL':float(np.mean([x['KL'] for x in v])),
                     'mean_gap':float(np.mean([x['gap'] for x in v])),
                     'gap_delta':cluster_interval([x['gap']-r['conditions']['clean']['gap'] for r,x in zip(group,v)],C.SEED)}
        spec=ops['head_full']['net_loss']-ops['matched']['net_loss'];h=ops['head_full']
        eligible=gate and len(group)>=minimum
        report[name]={'n':len(group),'eligible':eligible,'operations':ops,'loss_specificity':spec,
                      'passed':bool(eligible and h['net_loss']>=.10 and h['mcnemar_exact_p']<.05 and spec>=.05),
                      'interaction_gap':cluster_interval([r['conditions']['kv_replay']['gap']-r['conditions']['k_only']['gap']-r['conditions']['v_only']['gap']+r['conditions']['clean']['gap'] for r in group],C.SEED)}
    return report


def main(part):
    dest=C.OUT/f'{part}-result.json'
    if dest.exists():return
    corpus,prereg=frozen();formal=C.read(C.OUT/'formal-behavior.json');profile=formal['profile']
    chosen={b:int(C.digest([C.SEED,b,'causal'])[:8],16)%2 for b in range(32,288)}
    selected=[r for r in corpus if r['b'] in chosen and r['profile']==profile and r['t']==chosen[r['b']]];assert len(selected)==256
    plan={**C.identity(),'schema':'mlp_kv_mediation_execution.v1','part':part,'source_sha256':C.file_sha(__file__),
          'prereg_sha256':C.file_sha(C.OUT/'prereg.json'),'formal_sha256':C.file_sha(C.OUT/'formal-behavior.json'),
          'behavior_gate':formal['passed'],'profile':profile,'estimand':'conditional on behavior gate for task claims; exact transport is a separate computational check'}
    C.save(C.OUT/f'{part}-execution-prereg.json',plan)
    torch.set_num_threads(4);start=time.time()
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        if part=='mediation':result=mediate(model,tok,eos,selected,formal,prereg)
        elif part=='earlier':
            controls=[r for r in corpus if r['split']=='pilot' and r['profile']==profile][:8]
            result=earlier(model,tok,eos,[r for r in selected if r['split']=='lockbox'][:32])
            result['negative_controls']=negative(model,tok,eos,controls)
        else:result=capability(model,tok)
    C.save(dest,{**plan,'status':'complete',**result,'seconds':time.time()-start})
    print(part,{'seconds':round(time.time()-start,1),'rows':len(result['rows']),'checks':result.get('checks')},flush=True)


def mediate(model,tok,eos,items,formal,prereg):
    rows=[];scopes=[];native=defaultdict(list);words=word_indices(tok)
    for lo in range(0,len(items),4):
        batch=items[lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
        clean,raw0=prefix(model,x)
        edit,raw1=prefix(model,x,heads=[(34,2)])
        half,_=prefix(model,x,heads=[(34,2)],alpha=.5)
        scope=cache_changes(clean,edit,{35});scopes.append(scope)
        original=take(clean,[35]);replacement=take(edit,[35]);scrambled=control(original,replacement)
        cond={'clean':branch(model,x,clean,eos),'head_full':branch(model,x,edit,eos),'head_half':branch(model,x,half,eos),
              'kv_replay':branch(model,x,clean,eos,replacement),'kv_rescue':branch(model,x,edit,eos,original),
              'k_only':branch(model,x,clean,eos,{(35,'K'):replacement[35,'K']}),
              'v_only':branch(model,x,clean,eos,{(35,'V'):replacement[35,'V']}),
              'matched':branch(model,x,clean,eos,scrambled)}
        replay=equal_rows(cond['head_full'][0],cond['kv_replay'][0]);rescue=equal_rows(cond['clean'][0],cond['kv_rescue'][0])
        records=metrics(batch,cond,tok,words)
        delta_norm={k:(replacement[35,k].float()-original[35,k].float()).norm(dim=-1).cpu().tolist() for k in ['K','V']}
        norm_ratio={k:((scrambled[35,k].float()-original[35,k].float()).flatten(1).norm(dim=-1)/
                        (replacement[35,k].float()-original[35,k].float()).flatten(1).norm(dim=-1).clamp_min(1e-30)).cpu().tolist() for k in ['K','V']}
        for j,row in enumerate(records):
            row.update(index=lo+j,replay_logits_bitwise_equal=replay[j],rescue_logits_bitwise_equal=rescue[j],
                       replay_generation_equal=cond['head_full'][1][j]==cond['kv_replay'][1][j],
                       rescue_generation_equal=cond['clean'][1][j]==cond['kv_rescue'][1][j],
                       delta_KV_norm_by_GQA={k:v[j] for k,v in delta_norm.items()},scrambled_norm_ratio= {k:v[j] for k,v in norm_ratio.items()})
        rows.extend(records)
        for name,source in [('clean',original),('edited',replacement)]:
            for k in ['K','V']:native[f'{name}_{k}'].append(bit_array(source[35,k]))
        for name,values in [('clean',raw0),('edited',raw1)]:native[f'{name}_incoming_b35'].append(bit_array(values['incoming_b35']))
        native['clean_head34_2'].append(bit_array(raw0['head34_2']))
        if (lo+len(batch))%32==0:print('mediation',lo+len(batch),'/',len(items),flush=True)
    fields={k:np.concatenate(v) for k,v in native.items()};path=C.OUT/'native-slots.npz';np.savez(path,**fields)
    checks={k:sum(r[k] for r in rows) for k in ['replay_logits_bitwise_equal','rescue_logits_bitwise_equal','replay_generation_equal','rescue_generation_equal']}
    checks.update(only_B35_and_source=all(s['only_allowed_layers'] and s['all_earlier_positions_bitwise_equal'] for s in scopes),
                  nonzero_KV_cases=sum(any(x>0 for v in r['delta_KV_norm_by_GQA'].values() for x in v) for r in rows),
                  nonzero_future_distribution_cases=sum(r['conditions']['head_full']['KL']>0 for r in rows))
    return {'rows':rows,'checks':checks,'scopes':scopes,'summary':summarize(rows,formal['passed'],prereg['power_plan']['all_cases']['n_required']),
            'native':{'path':path.name,'sha256':C.file_sha(path),'dtype':'BF16 stored as uint16','fields':{k:list(v.shape) for k,v in fields.items()}}}


def earlier(model,tok,eos,items):
    rows=[];scopes=[];words=word_indices(tok)
    for lo in range(0,len(items),4):
        batch=items[lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
        clean,_=prefix(model,x);ablate,_=prefix(model,x,mlps=[33])
        scope=cache_changes(clean,ablate,{34,35});scopes.append(scope)
        a=take(clean,[34,35]);b=take(ablate,[34,35]);out={'clean':branch(model,x,clean,eos),'B33_MLP':branch(model,x,ablate,eos),
           'replay_both':branch(model,x,clean,eos,b),'rescue_both':branch(model,x,ablate,eos,a),
           'only_B34':branch(model,x,clean,eos,{k:v for k,v in b.items() if k[0]==34}),
           'only_B35':branch(model,x,clean,eos,{k:v for k,v in b.items() if k[0]==35})}
        for j,r in enumerate(metrics(batch,out,tok,words)):
            r.update(index=lo+j,replay_logits_bitwise_equal=bool(torch.equal(out['B33_MLP'][0][j],out['replay_both'][0][j])),
                     rescue_logits_bitwise_equal=bool(torch.equal(out['clean'][0][j],out['rescue_both'][0][j])))
            rows.append(r)
    checks={k:sum(r[k] for r in rows) for k in ['replay_logits_bitwise_equal','rescue_logits_bitwise_equal']}
    checks['only_B34_B35_and_source']=all(s['only_allowed_layers'] and s['all_earlier_positions_bitwise_equal'] for s in scopes)
    ops={name:{'mean_KL':float(np.mean([r['conditions'][name]['KL'] for r in rows])),
               'accuracy':float(np.mean([r['conditions'][name]['correct'] for r in rows]))} for name in rows[0]['conditions']}
    return {'rows':rows,'checks':checks,'scopes':scopes,'summary':{'operations':ops,'n':len(rows),'evidence':'engineering and descriptive endpoint; n32'}}


def negative(model,tok,eos,items):
    checks=[]
    for lo in range(0,len(items),4):
        batch=items[lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
        clean,raw0=prefix(model,x);neg,rawneg=prefix(model,x,heads=[(35,25)])
        cache_equal=not cache_changes(clean,neg,set())['slots']
        baseline=branch(model,x,clean,eos);edited_out=branch(model,x,neg,eos)
        for j,r in enumerate(batch):
            checks.append({'id':r['id'],'cache_bitwise_equal':cache_equal,
                'logits_bitwise_equal':bool(torch.equal(baseline[0][j],edited_out[0][j])),
                'generation_equal':baseline[1][j]==edited_out[1][j],
                'local_residual_delta_L2':float((rawneg['pre_mlp_b35'][j]-raw0['pre_mlp_b35'][j]).norm())})
    return checks


def capability(model,tok):
    inputs=C.read(C.CAP);assert sum(r['gate_eligible'] for r in inputs)==2200
    letters=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCDE'];rows=[]
    offset=42  # 1034-token prompt, source index 991.
    for lo in range(0,len(inputs),12):
        batch=inputs[lo:lo+12];prompts=[]
        for r in batch:
            text=r['question']+'\n'+'\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。'
            prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
        data={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()}
        size=data['input_ids'].shape[1];position=torch.tensor([max(size-len(p),size-1-offset) for p in prompts],device='cuda');idx=torch.arange(len(batch),device='cuda')
        predictions={}
        for name,heads,mlps in [('clean',[],[]),('B34_H2',[(34,2)],[]),('B33_MLP',[],[33])]:
            with edited(model,idx,position,heads,mlps,1.,False):logits=model(**data,use_cache=False,logits_to_keep=1).logits[:,-1]
            predictions[name]=['ABCDE'[int(logits[j,letters[:len(r['choices'])]].argmax())] for j,r in enumerate(batch)]
        for j,r in enumerate(batch):
            rows.append({'index':lo+j,'id':r.get('id'),'source':r.get('source'),'gate_eligible':r['gate_eligible'],'expected':r['answer'],
                         'actual_offset':int(size-1-position[j]),'conditions':{name:{'predicted':v[j],'correct':v[j]==r['answer']} for name,v in predictions.items()}})
        if (lo+len(batch))%240==0:print('capability',lo+len(batch),'/',len(inputs),flush=True)
    mask=np.asarray([r['gate_eligible'] for r in rows],bool);clean=[r['conditions']['clean']['correct'] for r in rows]
    return {'rows':rows,'summary':{name:capability_interval(clean,[r['conditions'][name]['correct'] for r in rows],mask) for name in ['B34_H2','B33_MLP']},
            'reference_only_n':int((~mask).sum()),'offset':offset,'input_sha256':C.file_sha(C.CAP)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['mediation','earlier','capability']);main(p.parse_args().part)
