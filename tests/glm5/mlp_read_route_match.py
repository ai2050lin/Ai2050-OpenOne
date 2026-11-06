"""Match each control vector to the actual corresponding joint perturbation."""
from collections import defaultdict
from contextlib import contextmanager
import time

import numpy as np
import torch

import mlp_read_route_common as C
from mlp_read_route_probe import batch_ids, inputs, positions, token_indices
from mlp_read_route_causal import branch, prefix, summarize, writers
from mlp_kv_mediation import metrics
from mlp_kv_behavior import word_indices
from mlp_fidelity_v2_tail import capability_interval
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease


@contextmanager
def matched_writers(model,pos,heads,mlps,reference):
    device=model.model.layers[34].mlp.down_proj.weight.device;i,j=token_indices(pos,device)
    handles=[];audit={};indices=list(zip(i.cpu().tolist(),j.cpu().tolist()))
    def modify(source,key):
        label=f'{key[0]}:{key[1]}';mapping=reference.get(label,{})
        rng=np.random.default_rng(C.SEED+key[0]*37+key[1]);n=source.shape[-1]
        perm=torch.as_tensor(rng.permutation(n),device=device)
        sign=torch.as_tensor(rng.choice(np.asarray([-1,1],np.float32),size=n),device=device)
        v=source.float()[:,perm]*sign
        norm=v.norm(dim=-1);fallback=norm.cpu().tolist()
        target=torch.tensor([mapping.get(k,fallback[p]) for p,k in enumerate(indices)],device=device)
        delta=v*(target/norm.clamp_min(1e-30))[:,None]
        zero=norm==0
        if zero.any():delta[zero,0]=target[zero]
        value=(source.float()-delta).to(source.dtype)
        audit[label]=[{'batch':b,'position':p,'reference_L2':t,'before_rounding_L2':a,'after_rounding_L2':z,
                       'enrolled_reference':(b,p) in mapping} for (b,p),t,a,z in zip(indices,target.cpu().tolist(),delta.norm(dim=-1).cpu().tolist(),
                                                                                       (value.float()-source.float()).norm(dim=-1).cpu().tolist())]
        return value
    try:
        for b,h in heads:
            sl=slice(h*128,(h+1)*128)
            def edit(_,args,b=b,h=h,sl=sl):
                x=args[0];out=x.clone();out[i,j,sl]=modify(x[i,j,sl],(b,h));return (out,*args[1:])
            handles.append(model.model.layers[b].self_attn.o_proj.register_forward_pre_hook(edit))
        for b in mlps:
            def edit(_,args,x,b=b):
                out=x.clone();out[i,j]=modify(x[i,j],(b,32));return out
            handles.append(model.model.layers[b].mlp.register_forward_hook(edit))
        yield audit
    finally:
        for h in handles:h.remove()


def grouped_reference(rows):
    ref=defaultdict(dict)
    for j,row in enumerate(rows):
        if row is None:continue
        for k,values in row['writer_norm']['joint'].items():
            for v in values:ref[k][j,v['source_position']]=v['applied_L2']
    return ref


def replay_match(model,tok,eos,route,result):
    source,prereg=inputs();pool=[r for r in source if r['split']!='discovery'];word_ids=word_indices(tok)
    screen=C.read_gz(C.OUT/'screen.json.gz');idx=screen['enrolled'];rows=result['rows'];heads,mlps=route['heads'],route['mlps']
    expected={r['screen_index']:r for r in rows};correction={v['screen_index']:v['screen_batch_ids'] for v in result['layout_diagnostics']}
    by_id={r['id']:i for i,r in enumerate(pool)};seen=set();all_audit=[];exact=0

    def run(batch_indices,active):
        nonlocal exact
        batch=[pool[i] for i in batch_indices];x=batch_ids(batch);pos=positions(batch,route['roles'])
        refs=[expected[i] if i in active else None for i in batch_indices]
        clean,_=prefix(model,x,pos)
        with matched_writers(model,pos,heads,mlps,grouped_reference(refs)) as audit:
            controlled=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
        comparisons=metrics(batch,{'clean':branch(model,x,clean,eos),'matched':branch(model,x,controlled,eos)},tok,word_ids)
        for j,i in enumerate(batch_indices):
            if i not in active:continue
            ref=expected[i];before=ref['conditions']['clean'];now=comparisons[j]['conditions']['clean']
            exact+=int((before['word'],before['gap'])==(now['word'],now['gap']));seen.add(i)
            ref['conditions']['matched_contextual_v1']=ref['conditions']['matched']
            ref['conditions']['matched']=comparisons[j]['conditions']['matched']
            entry={k:[v for v in values if v['batch']==j] for k,values in audit.items()}
            ref['strict_norm_audit']=entry
            all_audit.extend(v for values in entry.values() for v in values)

    for lo in range(0,len(idx),4):
        real=idx[lo:lo+4];pad=[i for i in range(4) if i not in real][:4-len(real)]
        active=set(real)-set(correction)
        if active:run(real+pad,active)
    for i,original in correction.items():run([by_id[x] for x in original],{i})
    assert len(seen)==len(idx) and all(v['enrolled_reference'] for v in all_audit)
    pre=[abs(v['reference_L2']-v['before_rounding_L2'])/max(v['reference_L2'],1e-30) for v in all_audit]
    post=[abs(v['reference_L2']-v['after_rounding_L2'])/max(v['reference_L2'],1e-30) for v in all_audit]
    norm={'vectors':len(pre),'max_pre_rounding_relative_error':max(pre),'median_post_rounding_relative_error':float(np.median(post)),
          'max_post_rounding_relative_error':max(post),'clean_recheck_exact':exact}
    checks={**result['checks'],'strict_matched_control':norm}
    summary=summarize(rows,checks['screen_exact']==len(rows) and exact==len(rows) and max(pre)<1e-5,prereg['power_plan']['n_required'])
    return {**result,'rows':rows,'checks':checks,'summary':summary}


def monitor(model,tok,route,original):
    source=C.read(C.CAP);heads,mlps=route['heads'],route['mlps'];letters=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCDE']
    records=original['rows'];max_error=0.;rechecked=0
    for lo in range(0,len(source),12):
        batch=source[lo:lo+12];prompts=[]
        for r in batch:
            text=r['question']+'\n'+'\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。'
            prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,
                        enable_thinking=False),add_special_tokens=False))
        data={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()}
        size=data['input_ids'].shape[1];pos=[sorted({max(size-len(p),size-1-d) for d in route['capability_offsets']}) for p in prompts]
        with writers(model,pos,heads,mlps,1.) as observed:joint=model(**data,use_cache=False,logits_to_keep=1).logits[:,-1]
        ref={k:{(b,p):n for b,p,n in zip(v['batch_index'],v['source_position'],v['applied_L2'])} for k,v in observed.items()}
        with matched_writers(model,pos,heads,mlps,ref) as audit:logits=model(**data,use_cache=False,logits_to_keep=1).logits[:,-1]
        for values in audit.values():
            max_error=max(max_error,max(abs(v['reference_L2']-v['before_rounding_L2'])/max(v['reference_L2'],1e-30) for v in values))
        for j,r in enumerate(batch):
            ids=letters[:len(r['choices'])];pred='ABCDE'[int(logits[j,ids].argmax())]
            clean_joint='ABCDE'[int(joint[j,ids].argmax())]
            old=records[lo+j];assert old['id']==r.get('id') and old['actual_offsets']==[size-1-v for v in pos[j]]
            rechecked+=int(clean_joint==old['conditions']['joint']['predicted'])
            old['conditions']['matched_contextual_v1']=old['conditions']['matched'];old['conditions']['matched']={'predicted':pred,'correct':pred==r['answer']}
        if (lo+len(batch))%240==0:print('strict ability',lo+len(batch),'/',len(source),flush=True)
    assert rechecked==len(source) and max_error<1e-5
    mask=np.asarray([r['gate_eligible'] for r in records]);base=[r['conditions']['clean']['correct'] for r in records]
    summary={k:capability_interval(base,[r['conditions'][k]['correct'] for r in records],mask) for k in ['joint','matched']}
    return {**original,'rows':records,'summary':summary,'joint_rechecked':rechecked,'max_norm_match_relative_error':max_error}


def main():
    out=C.OUT;plan=out/'strict-match-prereg.json'
    C.save(plan,{**C.identity(),'schema':'mlp_early_token_strict_control.v2','source_sha256':C.file_sha(__file__),
                 'frozen_route_sha256':C.file_sha(out/'route-freeze.json'),'no_threshold_change':True,
                 'rule':'Each randomized write uses the matching real joint branch applied L2 at the same sample, native writer and source position as its pre-BF16 norm. Store after-rounding error. Retain both controls and all original outcomes. Recheck clean or joint under identical batch layout.'})
    route=C.read(out/'route-freeze.json');torch.set_num_threads(4);start=time.time()
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        target=out/'causal-result-strict-v2.json.gz'
        if not target.exists():
            original=out/'causal-result.json.gz';r=C.read_gz(original)
            doc=replay_match(model,tok,eos,route,r) if r['rows'] else r
            C.write_gz(target,{**doc,'strict_match_prereg_sha256':C.file_sha(plan),'initial_causal_sha256':C.file_sha(original)})
            print('strict causal',doc['checks'],flush=True)
        target=out/'capability-result-strict-v2.json.gz'
        if not target.exists():
            original=out/'capability-result.json.gz';doc=monitor(model,tok,route,C.read_gz(original))
            C.write_gz(target,{**doc,'strict_match_prereg_sha256':C.file_sha(plan),'initial_capability_sha256':C.file_sha(original)})
            print('strict ability',doc['summary'],'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
