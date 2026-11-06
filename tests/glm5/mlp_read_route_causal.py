"""Native early-token writer intervention, precise cache replay and heldout read checks."""
import argparse
from contextlib import contextmanager
import time

import numpy as np
import torch
from scipy.stats import binomtest

import mlp_read_route_common as C
from mlp_read_route_probe import LAYERS, answer_read, batch_ids, cache_patch, gram, inputs, positions, read_statistics, token_indices
from mlp_kv_behavior import word_indices
from mlp_kv_mediation import equal_rows, metrics
from mlp_fidelity_v2_collect_v4 import greedy
from mlp_fidelity_v2_tail import capability_interval, cluster_interval
from mlp_functional_trace import bit_array, load_model
from shared.dist_protocol.mechanism_trace import GpuLease


@contextmanager
def writers(model,pos,heads,mlps,alpha,matched=False):
    device=model.model.layers[34].mlp.down_proj.weight.device;i,j=token_indices(pos,device)
    handles=[];norms={}
    def modify(source,key):
        delta=source.float()*alpha
        norm=delta.norm(dim=-1)
        if matched:
            rng=np.random.default_rng(C.SEED+key[0]*37+key[1]);n=delta.shape[-1]
            ix=torch.as_tensor(rng.permutation(n),device=device)
            sign=torch.as_tensor(rng.choice(np.asarray([-1,1],np.float32),size=n),device=device)
            delta=delta[:,ix]*sign
        value=(source.float()-delta).to(source.dtype)
        norms[f'{key[0]}:{key[1]}']={'batch_index':i.cpu().tolist(),'source_position':j.cpu().tolist(),
                                      'intended_L2':norm.cpu().tolist(),'applied_L2':(value.float()-source.float()).norm(dim=-1).cpu().tolist(),
                                      'pre_rounding_L2':delta.norm(dim=-1).cpu().tolist()}
        return value
    try:
        for b,h in heads:
            sl=slice(h*128,(h+1)*128)
            def replace(_,args,b=b,h=h,sl=sl):
                x=args[0];out=x.clone();out[i,j,sl]=modify(x[i,j,sl],(b,h));return (out,*args[1:])
            handles.append(model.model.layers[b].self_attn.o_proj.register_forward_pre_hook(replace))
        for b in mlps:
            def replace(_,args,value,b=b):
                out=value.clone();out[i,j]=modify(value[i,j],(b,32));return out
            handles.append(model.model.layers[b].mlp.register_forward_hook(replace))
        yield norms
    finally:
        for h in handles:h.remove()


def prefix(model,x,pos,heads=(),mlps=(),alpha=1.,matched=False):
    with writers(model,pos,heads,mlps,alpha,matched) as norms:
        cache=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
    return cache,norms


def branch(model,x,cache,eos,patches=()):
    with cache_patch(cache,patches):
        out=model(input_ids=x[:,C.CUT:],past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits=out.logits[:,-1].float();tokens=greedy(model,out.past_key_values,logits,eos,limit=4)
    return logits,tokens


def changed(clean,edited):
    forward=[];rescue=[];by_layer={}
    for b,(a,z) in enumerate(zip(clean.layers,edited.layers)):
        for kind in ['keys','values']:
            x=getattr(a,kind);y=getattr(z,kind)
            if torch.equal(x,y):continue
            diff=(x!=y).any(dim=-1).any(dim=1);begin=int(diff.nonzero()[:,1].min())
            ix=(slice(None),slice(None),slice(begin,C.CUT),slice(None))
            forward.append((b,kind,ix,y[ix].clone()));rescue.append((b,kind,ix,x[ix].clone()))
            by_layer[b,kind]=diff
    return forward,rescue,by_layer


def selected_slots(cache,pos,layers,kinds=('keys','values'),zero=False):
    first=cache.layers[0].keys;i,j=token_indices(pos,first.device);ix=(i,slice(None),j,slice(None));edits=[]
    for b in layers:
        for kind in kinds:
            value=getattr(cache.layers[b],kind)[ix]
            edits.append((b,kind,ix,torch.zeros_like(value) if zero else value.clone()))
    return edits


def direct_read(model,x,cache,pos,baseline):
    observed={}
    for b in LAYERS:
        patch=selected_slots(cache,pos,[b],kinds=('values',),zero=True)
        _,after=answer_read(model,x,cache,patch,[b]);a=baseline[b];z=after[b]
        prediction=[]
        for j,ps in enumerate(pos):
            index=torch.tensor(ps,device=x.device);w=a['weights'][j].index_select(-1,index)
            v=a['v'][j].index_select(-2,index);prediction.append(-torch.einsum('hp,hpd->hd',w,v))
        prediction=torch.stack(prediction).flatten(1);actual=(z['z'].float()-a['z'].float()).flatten(1)
        scale=prediction.norm(dim=-1)
        observed[b]={'Q_unchanged':equal_rows(a['q'].flatten(1),z['q'].flatten(1)),
                     'actual_delta_L2':actual.norm(dim=-1).cpu().tolist(),'predicted_delta_L2':scale.cpu().tolist(),
                     'relative_error':((actual-prediction).norm(dim=-1)/scale.clamp_min(1e-30)).cpu().tolist(),
                     'cosine':torch.nn.functional.cosine_similarity(actual,prediction,dim=-1).cpu().tolist()}
    return observed


def native_case(cache,j,ps):
    return {str(b):{kind:[bit_array(getattr(cache.layers[b],kind)[j,:,p]).tolist() for p in ps]
                    for kind in ['keys','values']} for b in LAYERS}


def execute_batch(model,tok,eos,rows,route,metric,words):
    x=batch_ids(rows);pos=positions(rows,route['roles']);heads,mlps=route['heads'],route['mlps']
    clean,_=prefix(model,x,pos);full,native_norm=prefix(model,x,pos,heads,mlps)
    delta,restore,changed_slots=changed(clean,full);layers=sorted({b for b,_ in changed_slots})
    raw=[{'positions':ps,'clean':native_case(clean,j,ps),'edited':native_case(full,j,ps)} for j,ps in enumerate(pos)]
    source_only=selected_slots(full,pos,layers);rest_only=delta+selected_slots(clean,pos,layers)
    cond={'clean':branch(model,x,clean,eos),'joint_1':branch(model,x,full,eos),
          'kv_replay':branch(model,x,clean,eos,delta),'kv_rescue':branch(model,x,full,eos,restore),
          'source_only':branch(model,x,clean,eos,source_only),'other_slots_only':branch(model,x,clean,eos,rest_only)}
    ctrl_norm=None
    for key,h,m,a,matched in [('joint_0.5',heads,mlps,.5,False),('head_only',heads,[],1.,False),
                               ('mlp_only',[],mlps,1.,False),('matched',heads,mlps,1.,True)]:
        edited,norm=prefix(model,x,pos,h,m,a,matched)
        cond[key]=branch(model,x,edited,eos)
        if matched:ctrl_norm=norm
        del edited
    clean_logits,fields=answer_read(model,x,clean);assert torch.equal(clean_logits,cond['clean'][0])
    observed=direct_read(model,x,clean,pos,fields);statistics=read_statistics(rows,fields,metric)
    weights=np.stack([fields[b]['weights'].cpu().numpy() for b in LAYERS],axis=1)
    reports=metrics(rows,cond,tok,words)
    replay=equal_rows(cond['joint_1'][0],cond['kv_replay'][0]);rescue=equal_rows(cond['clean'][0],cond['kv_rescue'][0])
    for j,r in enumerate(reports):
        def norms_for(norm):
            return {k:[{f:v[p] for f,v in data.items() if f!='batch_index'} for p,i in enumerate(data['batch_index']) if i==j] for k,data in norm.items()}
        r.update(replay_logits_bitwise_equal=replay[j],rescue_logits_bitwise_equal=rescue[j],
                 replay_generation_equal=cond['joint_1'][1][j]==cond['kv_replay'][1][j],rescue_generation_equal=cond['clean'][1][j]==cond['kv_rescue'][1][j],
                 changed_slots={f'{b}:{k}':v[j].nonzero().flatten().cpu().tolist() for (b,k),v in changed_slots.items()},
                 writer_norm={'joint':norms_for(native_norm),'matched':norms_for(ctrl_norm)},read=statistics[j],
                 read_verify={str(b):{k:v[j] for k,v in fields.items()} for b,fields in observed.items()})
    return reports,raw,weights


def summarize(rows,allowed,n_required):
    result={}
    groups={'all':rows,'confirm':[r for r in rows if r['split']=='confirm'],'lockbox':[r for r in rows if r['split']=='lockbox']}
    for key,group in groups.items():
        if not group:
            result[key]={'n':0,'eligible':False,'passed':False};continue
        base=np.asarray([r['conditions']['clean']['correct'] for r in group]);ops={}
        for op in group[0]['conditions']:
            compared=[r['conditions'][op] for r in group];correct=np.asarray([v['correct'] for v in compared])
            losses=int((base&~correct).sum());gains=int((~base&correct).sum())
            ops[op]={'accuracy':float(correct.mean()),'losses':losses,'gains':gains,'net_loss':float(base.mean()-correct.mean()),
                     'first_word_changes':sum(v['word']!=r['conditions']['clean']['word'] for v,r in zip(compared,group)),
                     'mcnemar_p':float(binomtest(losses,losses+gains).pvalue) if losses+gains else 1.,
                     'mean_KL':float(np.mean([v['KL'] for v in compared])),
                     'gap_delta':cluster_interval([v['gap']-r['conditions']['clean']['gap'] for v,r in zip(compared,group)],C.SEED)}
        h=ops['joint_1'];specific=h['net_loss']-ops['matched']['net_loss'];eligible=allowed and len(group)>=n_required
        result[key]={'n':len(group),'eligible':eligible,'passed':bool(eligible and h['net_loss']>=.1 and h['mcnemar_p']<.05 and specific>=.05),
                     'operations':ops,'specificity':specific,'median_clean_gap':float(np.median([r['conditions']['clean']['gap'] for r in group]))}
    return result


def causal(model,tok,eos,route):
    all_rows,prereg=inputs();screen=C.read_gz(C.OUT/'screen.json.gz')
    assert screen['route_sha256']==C.file_sha(C.OUT/'route-freeze.json')
    pool=[r for r in all_rows if r['split']!='discovery'];idx=screen['enrolled'];n=len(idx)
    if not n:return {'rows':[],'summary':{'all':{'n':0,'eligible':False,'passed':False}},'checks':{}}
    reports=[];raw=[];weights=[];word_ids=word_indices(tok);metric=gram(model)
    for lo in range(0,n,4):
        real=idx[lo:lo+4];pad=[i for i in range(4) if i not in real][:4-len(real)]
        batch=[pool[i] for i in real+pad];assert len(batch)==4
        info,slots,full=execute_batch(model,tok,eos,batch,route,metric,word_ids)
        reports.extend(info[:len(real)]);raw.extend(slots[:len(real)]);weights.extend(full[:len(real)])
        if (lo+len(real))%16==0:print('joint mediation',lo+len(real),'/',n,flush=True)
    def match(i,row):
        a=screen['rows'][i]['clean'];b=row['conditions']['clean']
        return (a['word'],a['gap'])==(b['word'],b['gap'])
    failures={j:i for j,i in enumerate(idx) if not match(i,reports[j])};layout=[]
    for j,i in failures.items():
        start=i//4*4;source=pool[start:start+4]
        info,slots,full=execute_batch(model,tok,eos,source,route,metric,word_ids);k=i-start
        layout.append({'screen_index':i,'initial':reports[j],'screen_batch_ids':[r['id'] for r in source],
                       'corrected':info[k],'screen':screen['rows'][i]['clean']})
        reports[j],raw[j],weights[j]=info[k],slots[k],full[k]
    matches=sum(match(i,r) for i,r in zip(idx,reports));assert len(reports)==n
    for j,r in enumerate(reports):r['index']=j;r['screen_index']=idx[j]
    arrays=C.OUT/'holdout-native.json.gz';C.write_gz(arrays,{'rows':[{'index':j,'id':reports[j]['id'],**v} for j,v in enumerate(raw)]})
    w=C.OUT/'holdout-weights.npz';np.savez_compressed(w,weights=np.asarray(weights,np.float32))
    checks={k:sum(r[k] for r in reports) for k in ['replay_logits_bitwise_equal','rescue_logits_bitwise_equal','replay_generation_equal','rescue_generation_equal']}
    checks.update(screen_exact=matches,layout_corrections=len(layout),only_B34_B35=all(all(k.startswith(('34:','35:')) for k in r['changed_slots']) for r in reports),
                  no_changes_before_earliest_origin=all(all(p>=min(raw[j]['positions']) for ps in r['changed_slots'].values() for p in ps) for j,r in enumerate(reports)),
                  nonzero_future_distribution_cases=sum(r['conditions']['joint_1']['KL']>0 for r in reports))
    read_summary={}
    for b in LAYERS:
        v=[r['read_verify'][str(b)] for r in reports]
        read_summary[str(b)]={'Q_unchanged':sum(z['Q_unchanged'] for z in v),'nonzero_read':sum(z['actual_delta_L2']>0 for z in v),
                              'median_cosine':float(np.median([z['cosine'] for z in v])),'median_relative_error':float(np.median([z['relative_error'] for z in v]))}
    return {'rows':reports,'summary':summarize(reports,matches==n,prereg['power_plan']['n_required']),
            'checks':checks,'layout_diagnostics':layout,'read_summary':read_summary,
            'native':{'raw_slots_sha256':C.file_sha(arrays),'full_weight_sha256':C.file_sha(w),'full_weight_shape':list(np.asarray(weights).shape)}}


def capability(model,tok,route):
    cases=C.read(C.CAP);assert sum(r['gate_eligible'] for r in cases)==2200
    letters=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCDE'];records=[]
    for lo in range(0,len(cases),12):
        batch=cases[lo:lo+12];prompts=[]
        for row in batch:
            text=row['question']+'\n'+'\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',row['choices']))+'\n请只回答正确选项字母。'
            prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,
                          enable_thinking=False),add_special_tokens=False))
        data={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()}
        size=data['input_ids'].shape[1];pos=[sorted({max(size-len(p),size-1-d) for d in route['capability_offsets']}) for p in prompts]
        scored={}
        for key,head,mlp,matched in [('clean',[],[],False),('joint',route['heads'],route['mlps'],False),('matched',route['heads'],route['mlps'],True)]:
            with writers(model,pos,head,mlp,1.,matched):logits=model(**data,use_cache=False,logits_to_keep=1).logits[:,-1]
            scored[key]=['ABCDE'[int(logits[j,letters[:len(r['choices'])]].argmax())] for j,r in enumerate(batch)]
        for j,r in enumerate(batch):
            records.append({'index':lo+j,'id':r.get('id'),'source':r.get('source'),'gate_eligible':r['gate_eligible'],'expected':r['answer'],
                            'actual_offsets':[size-1-v for v in pos[j]],'conditions':{k:{'predicted':v[j],'correct':v[j]==r['answer']} for k,v in scored.items()}})
        if (lo+len(batch))%240==0:print('capability',lo+len(batch),'/',len(cases),flush=True)
    mask=np.asarray([r['gate_eligible'] for r in records]);base=[r['conditions']['clean']['correct'] for r in records]
    return {'rows':records,'summary':{k:capability_interval(base,[r['conditions'][k]['correct'] for r in records],mask) for k in ['joint','matched']},
            'reference_only_n':int((~mask).sum()),'input_sha256':C.file_sha(C.CAP)}


def main(part):
    dest=C.OUT/f'{part}-result.json.gz'
    if dest.exists():return
    route=C.read(C.OUT/'route-freeze.json');torch.set_num_threads(4);start=time.time()
    freeze={**C.identity(),'schema':'mlp_early_token_intervention_execution.v1','part':part,'source_sha256':C.file_sha(__file__),
            'route_sha256':C.file_sha(C.OUT/'route-freeze.json'),'screen_sha256':C.file_sha(C.OUT/'screen.json.gz')}
    C.save(C.OUT/f'{part}-execution.json',freeze)
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        result=causal(model,tok,eos,route) if part=='causal' else capability(model,tok,route)
    C.write_gz(dest,{**freeze,'status':'complete','seconds':time.time()-start,**result})
    print(part,'n',len(result['rows']),'checks',result.get('checks'),'seconds',round(time.time()-start,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['causal','capability']);main(parser.parse_args().part)
