"""Independent NAME, printed ROW and physical slot with native Q/K replay."""
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import clt_b26_compact as common
from deploy.storage_policy import check_write, checked_dir
from source_binding_design import MODELS, WORDS, git, read, sha
from tools.power_analysis import check_phase

ROOT, HERE = common.ROOT, common.HERE
OUT = HERE/'result/clt_b26_three_cues_v1'
TMP = ROOT/'tests/glm5_temp/clt_b26_three_cues_v1'
SEED = 202611181529
SIZES = dict(pilot=32, discovery=64, confirm=128, intervention=128)
CELLS = tuple(itertools.product(range(2),range(2),range(3)))
RULES = ('name','label','position')
BUDGETS = (32,128,512)
METHODS = [('selected',n) for n in BUDGETS]+[('random',n) for n in BUDGETS]+[('mean',0)]
BLOCK, HEAD, KV = 26, 36, 7
QS, KS, CAP = common.HEAD_SLICE, common.KV_SLICE, common.CAP
PLAN = Path(__file__).with_suffix('.md')


def configure():
    common.OUT=OUT
    os.environ.update(TEMP=str(TMP),TMP=str(TMP),TMPDIR=str(TMP))


def previous_names():
    dirs=(OUT.parent/'conditional_lookup_native_v1',*OUT.parent.glob('clt_b26_*'))
    return {name for folder in dirs if folder!=OUT for p in folder.glob('*-materials.json')
            for row in read(p) for name in row['names']}


def make_family(tok,split,family,names,words,rng):
    labels=list(map(str,rng.permutation(list('ABCDE'))))
    interior=list(map(int,rng.permutation([0,1,2])))
    title=('Orion','Vega')[family%2]
    first=bool((family//2)%2==0)
    place='first' if first else 'last'
    tokens=[tok.encode(w,add_special_tokens=False) for w in words]
    assert all(len(c)==1 for c in tokens) and len({c[0] for c in tokens})==5
    rows=[]
    for cell,(label_swap,position_swap,rule) in enumerate(CELLS):
        printed=labels.copy()
        if label_swap:printed[1],printed[2]=printed[2],printed[1]
        outer=[3,4][::(-1 if position_swap else 1)]
        order=[outer[0],*interior,outer[1]]
        targets=[0,printed.index(labels[1]),order[0 if first else -1]]
        assert len(set(targets))==3 and targets[1]==1+label_swap
        content=('This archive contains five record lines. A printed ROW label is a letter and may appear in any order. '
                 'For physical position use the order of the record lines as displayed. '
                 'The old note supplies reference keys. Answer the final question with a single lowercase CODE word.\n'
                 f'Old note: NAME {names[0]}; printed ROW {labels[1]}; physical position {place}.\n'
                 f'Active Archive {title}:\n')
        spans={}
        for record in order:
            start=len(content);content+=f'ROW {printed[record]} | NAME {names[record]} | CODE {words[record]}'
            spans[record]=start,len(content);content+='\n'
        content+='Question: In the active archive, find the CODE'
        boundary=len(content)
        suffix=(f' in the record named {names[0]}',f' in the record with printed ROW {labels[1]}',f' in the {place} record of the table')[rule]
        content+=suffix+'. Return the code alone, with no punctuation.'
        prompt=tok.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:\n'
        encoding=tok(prompt,add_special_tokens=False,return_offsets_mapping=True); offset=prompt.index(content)
        offsets=encoding['offset_mapping']
        records=[[i for i,(a,b) in enumerate(offsets) if a<offset+spans[r][1] and b>offset+spans[r][0]] for r in range(5)]
        endpoint=max(i for i,(a,b) in enumerate(offsets) if a<b<=offset+boundary)
        assert all(records) and len(set(sum(records,[])))==len(sum(records,[]))
        target=targets[rule]
        rows.append(dict(id=f'{split}-{family:03}-{cell:02}',family=family,cell=cell,rule=rule,
                         label_swap=label_swap,position_swap=position_swap,first=first,title=title,names=names,codes=words,
                         printed=printed,physical_order=order,cue_targets=targets,target_record=target,
                         choice_ids=[c[0] for c in tokens],gold_id=tokens[target][0],records=records,endpoint=endpoint,
                         tokens=encoding['input_ids'],prompt=prompt))
    for a,b,c in zip(rows[::3],rows[1::3],rows[2::3]):
        e=a['endpoint']+1
        if not (len(a['tokens'])==len(b['tokens'])==len(c['tokens']) and a['endpoint']==b['endpoint']==c['endpoint']
                and a['tokens'][:e]==b['tokens'][:e]==c['tokens'][:e]):return None
    return rows


def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    checked_dir(OUT,128<<20,operation='Three cue campaign'); checked_dir(TMP)
    assert not (OUT/'prereg.json').exists()
    model=MODELS['qwen14']; revision=snapshot(model)
    prev=read(OUT.parent/'clt_b26_conflict_v7/prereg.json')
    assert revision==prev['provenance']['model_revision']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    rng=np.random.default_rng(SEED);used=previous_names();splits={}
    for split,count in SIZES.items():
        rows=[]
        for family in range(count):
            for _ in range(512):
                names=[''.join(rng.choice(list('bdfgklmnprstvz'))+rng.choice(list('aeiou')) for _ in range(3)).capitalize() for _ in range(5)]
                if len(set(names))!=5 or used.intersection(names): continue
                if any(len(tok.encode(' '+x,add_special_tokens=False))!=3 for x in names):continue
                batch=make_family(tok,split,family,names,list(map(str,rng.choice(WORDS,5,replace=False))),rng)
                if batch is not None:break
            else:raise RuntimeError('Unable to generate a length-matched family')
            rows.extend(batch);used.update(names)
        name=f'{split}-materials.json';common.commit(name,rows)
        splits[split]=dict(families=count,inputs=len(rows),sha=sha(OUT/name))
        print(f'Frozen {split}: {count} families, {len(rows)} inputs',flush=True)
    deps=[Path(__file__),PLAN,HERE/'clt_b26_compact.py',HERE/'source_binding_runtime.py',HERE/'source_binding_design.py',
          HERE/'source_binding_resources.py',ROOT/'shared/dist_protocol/natural_register_runner.py',
          ROOT/'shared/dist_protocol/mechanism_trace.py',ROOT/'tools/power_analysis.py',ROOT/'deploy/storage_policy.py',ROOT/'deploy/storage_policy.json']
    power=check_phase(.15,128,p_d=.25,alpha=.025);assert power['pass']
    common.commit('prereg.json',dict(utc=datetime.now(timezone.utc).isoformat(),code={f.relative_to(ROOT).as_posix():sha(f) for f in deps},
                  materials=splits,power=power,provenance=dict(git_sha=git('rev-parse','HEAD'),git_dirty=True,model_path=str(model),model_revision=revision,seed=SEED)))


def materials(split,pre):
    path=OUT/f'{split}-materials.json';assert sha(path)==pre['materials'][split]['sha']
    return read(path)


def seal():
    candidate=read(OUT/'candidate.json')
    assert candidate['prereg_sha']==sha(OUT/'prereg.json') and candidate['predictor_sha']==sha(OUT/'predictor.npz')
    return candidate


def grade(rt, logits, rows):
    scores=[]
    for i,row in enumerate(rows):
        values=logits[i,row['choice_ids']]; gold=row['target_record']; top=int(logits[i].argmax()); labels=[j for j in range(5) if j!=gold]
        scores.append(dict(id=row['id'],scores=values.cpu().tolist(),pred=int(values.argmax()),top1=top,
                           top1_correct=top==row['gold_id'],gold_margin=float(values[gold]-values[labels].max())))
    return scores


def behavior(rows,scores):
    assert [r['id'] for r in rows]==[s['id'] for s in scores]
    cells=[]
    for i in range(len(CELLS)):
        group=[s for r,s in zip(rows,scores,strict=True) if r['cell']==i]
        cells.append(dict(n=len(group),forced=sum(s['gold_margin']>0 for s in group),top1=sum(s['top1_correct'] for s in group)))
    totals={k:sum(g[k] for g in cells) for k in ('n','forced','top1')}
    passed=(all(min(g['forced'],g['top1'])/g['n']>=.90 for g in cells)
            and min(totals['forced'],totals['top1'])/totals['n']>=.95)
    return dict(cells=cells,totals=totals,passed=passed)


def collect(split):
    import torch as t
    import torch.nn.functional as F
    import transformers
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_runtime import Runtime
    from source_binding_resources import resources
    pre=common.checked()
    if split!='pilot':assert read(OUT/'pilot-collection.json')['passed']
    if split=='confirm':
        seal(); fit=np.load(OUT/'predictor.npz')
    rows=materials(split,pre); assert not (OUT/f'{split}-collection.json').exists()
    width=384; start=time.monotonic(); total=((len(rows)+width-1)//width)*40

    def progress(shard,b):
        n=shard*40+b+1;elapsed=time.monotonic()-start
        event=dict(split=split,shard=shard,block=b,completed=n,total=total,elapsed=elapsed,eta=(total-n)*elapsed/n if n else None,
                   cuda_peak=t.cuda.max_memory_allocated(),**resources())
        with (OUT/f'{split}-progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        assert elapsed<45*60 and (event.get('shared_gpu_bytes') or 0)<1<<30 and event['ram_available_bytes']>2<<30

    progress(0,-1);assert snapshot(MODELS['qwen14'])==pre['provenance']['model_revision']
    rt=Runtime('qwen14'); assert (rt.heads,rt.kv,rt.hd)==(40,8,128)
    t.backends.cuda.matmul.allow_tf32=False
    shards={}; all_scores=[]; errors=[]; counts=[]
    for shard,lower in enumerate(range(0,len(rows),width)):
        check_write(OUT,24<<20,operation='Three cues shard'); rr=rows[lower:lower+width]; name=f'{split}-{shard:02}'
        file=OUT/(name+'.npz'); result_file=OUT/(name+'.json')
        if result_file.exists():
            summary=read(result_file);assert sha(file)==summary['sha'] and summary['prereg_sha']==sha(OUT/'prereg.json')
            assert summary['candidate_sha']==(sha(OUT/'candidate.json') if split=='confirm' else None)
        else:
            saved={}; prefix=[]; moment=np.zeros((2,2,rt.dim),np.float64); number=np.zeros(2,np.int64)

            def put(key,value):saved.setdefault(key,[]).append(value.detach().cpu().numpy())

            def block(b):
                layer=Runtime.block(rt,b)
                if b!=BLOCK:return layer
                att=layer.self_attn
                W={'q':att.q_proj.weight[QS].float(),'k':att.k_proj.weight[KS].float(),'o':att.o_proj.weight[:,QS].float()}
                s={'offset':0};compiled={}
                if split=='confirm':
                    for k in ('q','k'):
                        mu=t.tensor(fit['mean_'+k],device='cuda'); intercept=W[k]@mu
                        for method,n in METHODS:
                            idx=t.tensor(fit[f'{k}_{method}_{n}'],device='cuda') if n else None
                            compiled[k,method,n]=(idx,mu,intercept,W[k][:,idx].contiguous() if n else None)

                def prepare(m,args,kw):s.update(z=kw['hidden_states'],mask=kw['attention_mask'][:,0,-1],pe=kw['position_embeddings'])
                def capture(k,region):
                    def hook(m,args,out):s[k]=out[...,region].contiguous()
                    return hook

                def route(q,k):
                    cos,sin=s['pe']; q=att.q_norm(q); k=att.k_norm(k)
                    q=q*cos[:,-1:]+rt.mod.rotate_half(q)*sin[:,-1:]
                    k=k*cos+rt.mod.rotate_half(k)*sin
                    logits=(q.float()*k.float()).sum(-1)*att.scaling
                    alpha=logits.masked_fill(~s['mask'],float('-inf')).softmax(-1)
                    return (s['spans']*alpha[:,None]).sum(-1),(alpha[...,None]*s['v'].float()).sum(1)

                def before_o(m,args):
                    z=s['z'];n,T=z.shape[:2]; batch=rr[s['offset']:s['offset']+n]; s['offset']+=n
                    spans=t.zeros((n,5,T),device='cuda')
                    for j,r in enumerate(batch):
                        p=T-len(r['tokens'])
                        for k,sites in enumerate(r['records']):spans[j,k,[p+i for i in sites]]=1
                    s['spans']=spans
                    q=s['q'][:,-1:]; k=s['k']; mass,h=route(q,k); native=args[0][:,-1,QS].float()
                    put('mass_actual',mass);put('head_actual',native);put('sdpa',(h-native).abs().amax(-1))
                    if split=='pilot':return
                    if split=='discovery':
                        for j,part in enumerate((z[:,-1],z[s['mask']])):
                            x=part.double();moment[j,0]+=x.sum(0).cpu().numpy();moment[j,1]+=x.square().sum(0).cpu().numpy();number[j]+=len(x)
                        put('z_final',z[:,-1].contiguous().view(t.uint16));return
                    probe=t.tensor(fit['q_selected_32'],device='cuda')
                    put('probe',z[:,-1,probe].float())
                    residual=[]; labels=[]
                    for method,n in METHODS:
                        compact={}
                        for key in ('q','k'):
                            idx,mu,intercept,w=compiled[key,method,n]; x=z[:,-1:] if key=='q' else z
                            p=F.linear(x[...,idx].float()-mu[idx],w)+intercept if n else intercept.expand(*x.shape[:-1],-1)
                            compact[key]=p.bfloat16()
                        for branch in ('q','k','joint'):
                            label=f'{branch}_{method}_{n}'; actual_q=compact['q'] if branch!='k' else q; actual_k=compact['k'] if branch!='q' else k
                            mass,head=route(actual_q,actual_k)
                            put('mass_'+label,mass)
                            delta=head.bfloat16().float()-native
                            if n==512 and method=='selected':put('head_delta_'+branch,delta)
                            residual.append(delta);labels.append(label)
                    values=F.linear(t.stack(residual,dim=1),W['o']).square().mean(-1)
                    for j,label in enumerate(labels):put('write_mse_'+label,values[:,j])

                att.register_forward_pre_hook(prepare,with_kwargs=True)
                att.q_proj.register_forward_hook(capture('q',QS));att.k_proj.register_forward_hook(capture('k',KS));att.v_proj.register_forward_hook(capture('v',KS))
                att.o_proj.register_forward_pre_hook(before_o)
                return layer

            def callback(b,layer,x,y,batch,T,offset=0,**kw):
                if b==BLOCK-1:
                    indices=[T-len(r['tokens'])+r['endpoint'] for r in batch]
                    prefix.append(y[t.arange(len(batch),device='cuda'),indices].cpu().contiguous().view(t.uint16).numpy())
                if b>=0 and offset+len(batch)==len(rr) and (b%5==4 or b==BLOCK):progress(shard,b)

            rt.block=block
            with GpuLease(),t.inference_mode():
                _,logits=rt.forward(rr,callback); scores=grade(rt,logits,rr);del logits
            data={k:np.concatenate(v) for k,v in saved.items()}
            if split=='discovery':data.update(moment=moment,number=number)
            a=np.concatenate(prefix).reshape(-1,3,rt.dim)
            equal=int((a[:,0,None]==a[:,1:]).all(axis=(1,2)).sum()); error=float(data['sdpa'].max())
            common.archive(file.name,data)
            summary=dict(sha=sha(file),scores=scores,prefix_equal=equal,max_sdpa=error,prereg_sha=sha(OUT/'prereg.json'),candidate_sha=sha(OUT/'candidate.json') if split=='confirm' else None)
            common.commit(result_file.name,summary)
        assert summary['max_sdpa']<=CAP and summary['prefix_equal']==len(rr)//3,(name,summary['max_sdpa'],summary['prefix_equal'])
        shards[file.name]=summary['sha'];all_scores.extend(summary['scores']);errors.append(summary['max_sdpa']);counts.append(summary['prefix_equal'])
    summary=behavior(rows,all_scores)
    common.commit(f'{split}-collection.json',dict(**summary,shards=shards,max_sdpa=max(errors),prefix_equal=sum(counts),elapsed=time.monotonic()-start,
                  provenance=pre['provenance'],torch=t.__version__,transformers=transformers.__version__))
    print(f'{split}: {summary["totals"]}; behavior={summary["passed"]}',flush=True)


def read_collection(split):
    collection=read(OUT/f'{split}-collection.json');data=[]
    for name,digest in sorted(collection['shards'].items()):
        assert sha(OUT/name)==digest
        with np.load(OUT/name) as z:data.append({k:z[k] for k in z.files})
    return collection,data


def select():
    from source_binding_runtime import Runtime
    pre=common.checked(); summary,blocks=read_collection('discovery');assert summary['passed']
    counts=np.sum([x['number'] for x in blocks],axis=0);mom=np.sum([x['moment'] for x in blocks],axis=0)
    mean=mom[:,0]/counts[:,None]; variance=np.maximum(mom[:,1]/counts[:,None]-mean**2,0)
    rt=Runtime('qwen14'); params={};rng=np.random.default_rng(SEED+57); prefix=f'model.layers.{BLOCK}.self_attn.'
    for j,k in enumerate(('q','k')):
        part=rt.weight(prefix+f'{k}_proj.weight','cpu')[QS if k=='q' else KS].contiguous()
        energy=variance[j]*np.square(part.float().numpy().astype(np.float64)).sum(0); idx=np.argsort(-energy,kind='stable');rand=rng.permutation(rt.dim)
        params.update({f'mean_{k}':mean[j].astype(np.float32),f'rank_{k}':energy,f'weight_{k}':part.view(rt.t.uint16).numpy()})
        for n in BUDGETS:params.update({f'{k}_selected_{n}':idx[:n],f'{k}_random_{n}':rand[:n]})
    w=rt.weight(prefix+'o_proj.weight','cpu')[:,QS].contiguous();params['weight_o']=w.view(rt.t.uint16).numpy()
    raw=np.concatenate([x['z_final'] for x in blocks]); z=((raw[:,params['q_selected_32']].astype(np.uint32)<<16).view(np.float32)).astype(np.float64)
    mu=z.mean(0); sd=np.maximum(z.std(0),1e-6); z=(z-mu)/sd
    rules=np.tile(np.array([c[-1] for c in CELLS]),SIZES['discovery']); target=np.eye(3)[rules]-1/3
    params.update(probe_mu=mu,probe_sd=sd,probe_coef=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@target))
    common.archive('predictor.npz',params)
    common.commit('candidate.json',dict(utc=datetime.now(timezone.utc).isoformat(),prereg_sha=sha(OUT/'prereg.json'),
                  discovery_sha=sha(OUT/'discovery-collection.json'),predictor_sha=sha(OUT/'predictor.npz'),provenance=pre['provenance']))
    print('Sealed Q/K coordinates, scalar slices and independent readout',flush=True)


def evaluate():
    pre=common.checked();seal(); c,data=read_collection('confirm'); assert c['passed'];rows=materials('confirm',pre)
    values={k:np.concatenate([x[k] for x in data]) for k in data[0]}; original=values['mass_actual']
    cues=np.array([r['cue_targets'] for r in rows]); goal=np.array([r['target_record'] for r in rows]); rule=np.array([r['rule'] for r in rows]); i=np.arange(len(rows))
    bootstrap=np.random.default_rng(SEED+61).integers(0,SIZES['confirm'],(10000,SIZES['confirm']))

    def ci(x):
        z=x.reshape(SIZES['confirm'],-1).mean(1)
        return [float(z.mean()),*np.quantile(z[bootstrap].mean(1),[.0125,.9875]).tolist()]

    def centered(mass):
        log=np.log(np.maximum(mass[i[:,None],cues],1e-20));return log-log.mean(1,keepdims=True)

    native=original.argmax(1); tri=original[i[:,None],cues].argmax(1)
    c0=centered(original); preference=1.5*c0[i,rule]
    diagnostic=[]
    for j in range(3):
        f=rule==j
        other=[float((native[f]==cues[f,k]).mean()) for k in range(3)]
        diagnostic.append(dict(rule=RULES[j],all_five_gold=float((native[f]==goal[f]).mean()),among_three_gold=float((tri[f]==j).mean()),
                               native_landing_by_cue=other,other_records=1-sum(other),preference=ci(preference[f])))
    by_cell=[dict(cell=list(cell),accuracy=float((native[j::len(CELLS)]==goal[j::len(CELLS)]).mean()),preference=ci(preference[j::len(CELLS)])) for j,cell in enumerate(CELLS)]
    switches=all(x['all_five_gold']>=.75 and x['preference'][1]>0 for x in diagnostic)
    # Exchange tests use immutable record identity as the index.
    own_mass=np.log(np.maximum(original[i,goal],1e-20)).reshape(-1,2,2,3)
    exchange={name:{RULES[j]:ci((own_mass[:,1,:,j]-own_mass[:,0,:,j]) if axis==0 else (own_mass[:,:,1,j]-own_mass[:,:,0,j]))
                    for j in range(3)} for axis,name in enumerate(('label_swap','position_swap'))}
    stats={};errors={}
    for branch in ('q','k','joint'):
        for method,n in METHODS:
            label=f'{branch}_{method}_{n}';pred=values['mass_'+label]; error=np.square(centered(pred)-c0).mean(1)
            errors[label]=error
            stats[label]=dict(five_way_agreement=float((pred.argmax(1)==native).mean()),cue_agreement=float((pred[i[:,None],cues].argmax(1)==tri).mean()),
                              margin_mse=float(error.mean()),write_mse=float(values['write_mse_'+label].mean()))
    gates={}
    for branch in ('q','k'):
        sel=f'{branch}_selected_512';compare={}
        for baseline in (f'{branch}_random_512',f'{branch}_mean_0'):
            interval=ci(errors[baseline]-errors[sel]); gain=1-stats[sel]['margin_mse']/max(stats[baseline]['margin_mse'],1e-20)
            compare[baseline]=dict(gain=gain,paired=interval,passed=bool(gain>=.2 and interval[1]>0))
        gates[branch]=dict(passed=bool(stats[sel]['five_way_agreement']>=.90 and all(q['passed'] for q in compare.values())),against=compare)
    with np.load(OUT/'predictor.npz') as fit:
        guess=(((values['probe']-fit['probe_mu'])/fit['probe_sd'])@fit['probe_coef']).argmax(1)
    result=dict(provenance=pre['provenance'],candidate_sha=sha(OUT/'candidate.json'),collection_sha=sha(OUT/'confirm-collection.json'),
                diagnostic=diagnostic,all_three_switch=switches,by_cell=by_cell,exchange=exchange,probe_accuracy=float((guess==rule).mean()),
                never_switch_baselines={RULES[j]:float((cues[:,j]==goal).mean()) for j in range(3)},prediction=stats,gates=gates,
                admitted=[k for k,g in gates.items() if g['passed']])
    common.commit('evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('prediction','by_cell')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    configure(); stage=sys.argv[1]
    if stage in ('pilot','discovery','confirm'):collect(stage)
    else:{'freeze':freeze,'select':select,'evaluate':evaluate}[stage]()
