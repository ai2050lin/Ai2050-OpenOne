"""Frozen identity-versus-row materials and native B26/H36 prediction."""
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import clt_b26_compact as shared
from deploy.storage_policy import check_write, checked_dir
from source_binding_design import MODELS, WORDS, git, read, sha
from tools.power_analysis import check_phase

ROOT, HERE = shared.ROOT, shared.HERE
OUT = HERE / 'result/clt_b26_conflict_v1'
TMP = ROOT / 'tests/glm5_temp/clt_b26_conflict_v1'
shared.OUT = OUT
os.environ.update(TEMP=str(TMP), TMP=str(TMP), TMPDIR=str(TMP))
commit, archive, checked = shared.commit, shared.archive, shared.checked
BLOCK, HEAD, KV = 26, 36, 7
QS, KS = shared.HEAD_SLICE, shared.KV_SLICE
CAP = shared.CAP
SEED = 202611181026
SIZES = dict(pilot=16, discovery=64, confirm=128, intervention=128)
CELLS = list(itertools.product(range(2), repeat=4))
BRANCHES = ('q', 'k', 'v')
ARMS = [(branch, method, n) for branch in (*BRANCHES, 'joint') for method in ('selected', 'random') for n in (32, 128)]
ARMS += [(branch, 'mean', 0) for branch in (*BRANCHES, 'joint')]


def make_family(tok, split, family, names, words):
    titles = ['Orion', 'Vega'][::1 if family % 2 == 0 else -1]
    codes = [tok.encode(w, add_special_tokens=False) for w in words]
    assert all(len(v) == 1 for v in codes)
    output = []
    for cell, (block, name_row, conflict, rule) in enumerate(CELLS):
        identity = (family // 2) % 2
        name_slot = 2*block + name_row
        row_slot = 2*block + (name_row ^ conflict)
        text = ('Two archives assign codes to names. Each archive has two numbered rows. '
                'Use the specified archive. Use only the lookup key indicated at the end. '
                'A name key matches the requested name. A row key matches the requested row number.\n')
        spans = []
        for b, title in enumerate(titles):
            text += f'Archive {title}:\n'
            first = identity ^ name_row if b == block else (family // 4) % 2
            for j in range(2):
                start = len(text)
                text += f'Row {j+1}: {names[first ^ j]} has code {words[2*b+j]}.'
                spans.append((start, len(text)))
                text += '\n'
        text += f'Archive to use: {titles[block]}\nRequested name: {names[identity]}\nRequested row: {1+(name_row^conflict)}\nLookup key to use:'
        boundary = len(text)
        text += (' name' if rule == 0 else ' row') + '\nReturn only the code word.'
        prompt = tok.apply_chat_template([dict(role='user', content=text)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + 'Answer:\n'
        tokenized = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offset = prompt.index(text)
        positions = [[i for i,(x,y) in enumerate(tokenized['offset_mapping']) if x < offset+b and y > offset+a] for a,b in spans]
        endpoint = max(i for i,(a,b) in enumerate(tokenized['offset_mapping']) if a < b and b <= boundary+offset)
        assert all(positions) and len(set(sum(positions, []))) == len(sum(positions, []))
        slot = (name_slot, row_slot)[rule]
        output.append(dict(id=f'{split}-{family:03}-{cell:02}', family=family, cell=cell, names=names,
                           prompt=prompt, tokens=tokenized['input_ids'], records=positions, endpoint=endpoint,
                           rule=rule, conflict=bool(conflict), name_slot=name_slot, row_slot=row_slot,
                           target_record=slot, gold_id=codes[slot][0], choice_ids=[v[0] for v in codes]))
    for a,b in zip(output[::2],output[1::2]):
        p = a['endpoint']+1
        assert a['endpoint'] == b['endpoint'] and a['tokens'][:p] == b['tokens'][:p]
    return output


def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    checked_dir(OUT, 240<<20, operation='CLT conflict campaign')
    checked_dir(TMP)
    assert not (OUT/'prereg.json').exists()
    model = MODELS['qwen14']
    revision = snapshot(model)
    assert revision == read(OUT.parent/'clt_b26_compact_v1/prereg.json')['provenance']['model_revision']
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    used = {name for folder in ('conditional_lookup_native_v1','clt_b26_native_v1','clt_b26_compact_v1')
            for f in (OUT.parent/folder).glob('*-materials.json') for row in read(f) for name in row['names']}
    rng = np.random.default_rng(SEED)
    splits = {}
    for split,count in SIZES.items():
        rows = []
        for i in range(count):
            while True:
                names = [''.join(rng.choice(list('bdfgklmnprstvz'))+rng.choice(list('aeiou')) for _ in range(3)).capitalize() for _ in range(2)]
                if len(set(names)) == 2 and not used.intersection(names) and len({len(tok.encode(' '+n,add_special_tokens=False)) for n in names}) == 1:
                    break
            used.update(names)
            rows.extend(make_family(tok, split, i, names, list(map(str,rng.choice(WORDS,4,replace=False)))))
        name = split+'-materials.json'
        commit(name, rows)
        splits[split] = dict(families=count, inputs=len(rows), sha=sha(OUT/name))
    deps = [Path(__file__), HERE/'clt_b26_conflict.md', HERE/'clt_b26_compact.py', HERE/'source_binding_runtime.py', HERE/'source_binding_design.py', HERE/'source_binding_resources.py', ROOT/'deploy/storage_policy.py', ROOT/'deploy/storage_policy.json', ROOT/'shared/dist_protocol/mechanism_trace.py', ROOT/'tools/power_analysis.py']
    power = check_phase(.15,128,p_d=.25,alpha=.025)
    assert power['pass']
    commit('prereg.json',dict(utc=datetime.now(timezone.utc).isoformat(), code={p.relative_to(ROOT).as_posix():sha(p) for p in deps}, materials=splits, power=power,
                              provenance=dict(model_path=str(model),model_revision=revision,seed=SEED,git_sha=git('rev-parse','HEAD'),git_dirty=True)))
    print('Frozen 336 families with four independent partitions',flush=True)


def materials(split, pre):
    file = OUT/f'{split}-materials.json'
    assert sha(file)==pre['materials'][split]['sha']
    return read(file)


def seal():
    x = read(OUT/'candidate.json')
    assert x['prereg_sha']==sha(OUT/'prereg.json') and x['predictor_sha']==sha(OUT/'predictor.npz')
    return x


def behavior(rows,scores):
    groups=[]
    for c in range(16):
        part=[s for r,s in zip(rows,scores,strict=True) if r['cell']==c and r['id']==s['id']]
        assert len(part)==len(rows)//16
        groups.append([sum(s['gold_margin']>0 for s in part),sum(s['top1_correct'] for s in part),len(part)])
    passed = all(min(a,b)/n>=.90 for a,b,n in groups) and min(sum(g[k] for g in groups) for k in (0,1))/len(rows)>=.95
    return groups,passed


def collect(split):
    import torch as t
    from torch.nn import functional as F
    import transformers
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_runtime import Runtime
    from source_binding_resources import resources

    pre=checked()
    if split!='pilot':
        assert read(OUT/'pilot-collection.json')['passed']
    if split=='confirm':
        candidate=seal()
        fit=np.load(OUT/'predictor.npz')
    rows=materials(split,pre)
    assert not (OUT/f'{split}-collection.json').exists()
    started=time.monotonic()
    total=((len(rows)+255)//256)*40
    all_scores,shards,pairs,technical=[],{},0,[]

    def progress(shard,block):
        n=shard*40+block+1
        elapsed=time.monotonic()-started
        record=dict(split=split,shard=shard,block=block,completed=n,total=total,elapsed=elapsed,eta=(total-n)*elapsed/n if n else None,
                    cuda_peak=t.cuda.max_memory_allocated(),**resources())
        with (OUT/f'{split}-progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        assert elapsed<45*60 and (record.get('shared_gpu_bytes') or 0)<1<<30

    progress(0,-1)
    assert snapshot(MODELS['qwen14'])==pre['provenance']['model_revision']
    rt=Runtime('qwen14')
    t.backends.cuda.matmul.allow_tf32=False
    for shard,start in enumerate(range(0,len(rows),256)):
        check_write(OUT,32<<20,operation='CLT conflict shard')
        rr=rows[start:start+256]
        name=f'{split}-{shard:02}'
        saved={}
        pre_prefix=[]
        mom=np.zeros((2,2,rt.dim),np.float64)
        counts=np.zeros(2,dtype=np.int64)
        sensitivity=np.zeros(rt.width,np.float64)

        def put(key,value):
            saved.setdefault(key,[]).append(value.detach().cpu().numpy())

        def block(b):
            layer=Runtime.block(rt,b)
            if b!=BLOCK:return layer
            att=layer.self_attn
            weights=dict(q=att.q_proj.weight[QS].float(),k=att.k_proj.weight[KS].float(),v=att.v_proj.weight[KS].float(),o=att.o_proj.weight[:,QS].float())
            s=dict(off=0)
            compiled={}
            if split=='confirm':
                units=t.tensor(candidate['units'],device='cuda')
                for k in BRANCHES:
                    mean=t.tensor(fit['mean_'+k],device='cuda')
                    for method,n in [('selected',32),('selected',128),('random',32),('random',128),('mean',0)]:
                        idx=t.tensor(fit[f'{k}_{method}_{n}'],device='cuda') if n else None
                        compiled[k,method,n]=(idx,mean,weights[k]@mean,weights[k][:,idx].contiguous() if n else None)
            else:units=None
            wg=layer.mlp.gate_proj.weight if units is None else layer.mlp.gate_proj.weight[units]
            wu=layer.mlp.up_proj.weight if units is None else layer.mlp.up_proj.weight[units]

            def prepare(m,args,kw):s.update(z=kw['hidden_states'],mask=kw['attention_mask'][:,0,-1],pe=kw['position_embeddings'])
            def capture(k,region):
                def hook(m,args,out):s[k]=out[...,region].contiguous()
                return hook

            def route(p):
                q,k=rt.mod.apply_rotary_pos_emb(att.q_norm(p['q']).unsqueeze(1),att.k_norm(p['k']).unsqueeze(1),*s['pe'])
                logits=(q[:,0,-1,None].float()*k[:,0].float()).sum(-1)*att.scaling
                alpha=logits.masked_fill(~s['mask'],float('-inf')).softmax(-1)
                return (s['records']*alpha[:,None]).sum(-1),(alpha[...,None]*p['v'].float()).sum(1)

            def down(write):
                z=layer.post_attention_layernorm(s['res']+(s['background'].float()+write).bfloat16())
                g,u=F.linear(z,wg),F.linear(z,wu)
                return (F.silu(g)*u).float()

            def before_o(m,args):
                z=s['z']; n,w=z.shape[:2]
                batch=rr[s['off']:s['off']+n]; s['off']+=n
                records=t.zeros((n,4,w),device='cuda')
                for i,r in enumerate(batch):
                    pad=w-len(r['tokens'])
                    for j,idx in enumerate(r['records']):records[i,j,[pad+x for x in idx]]=1
                s['records']=records
                p={k:s[k] for k in BRANCHES}
                mass,calc=route(p)
                raw=args[0][:,-1]
                put('mass_actual',mass)
                put('sdpa',(raw[:,QS].float()-calc).abs().amax(-1))
                if split=='pilot':return
                bg=raw.clone();bg[:,QS]=0
                s['background']=F.linear(bg,att.o_proj.weight)
                write=F.linear(raw[:,QS].float(),weights['o'])
                if split=='discovery':
                    for i,xx in enumerate((z[:,-1],z[s['mask']])):
                        d=xx.double(); mom[i,0]+=d.sum(0).cpu().numpy(); mom[i,1]+=d.square().sum(0).cpu().numpy(); counts[i]+=len(d)
                    s['zero']=down(t.zeros_like(write))
                    put('q_final',z[:,-1].contiguous().view(t.uint16))
                    return
                put('actual_head',raw[:,QS].float())
                put('q_final',z[:,-1].contiguous().view(t.uint16))
                predictions={}
                for k in BRANCHES:
                    xx=z[:,-1:] if k=='q' else z
                    for method,budget in [('selected',32),('selected',128),('random',32),('random',128),('mean',0)]:
                        idx,mu,bias,W=compiled[k,method,budget]
                        value=F.linear(xx[...,idx].float()-mu[idx],W)+bias if budget else bias.expand(*xx.shape[:-1],-1)
                        predictions[k,method,budget]=value.bfloat16()
                s['a_predictions']={}
                for k,method,budget in ARMS:
                    label=f'{k}_{method}_{budget}'
                    projected={j:predictions[j,method,budget] if k in (j,'joint') else p[j] for j in BRANCHES}
                    mass,h=route(projected)
                    h=h.bfloat16().float()
                    estimate=F.linear(h,weights['o'])
                    put('mass_'+label,mass)
                    put('head_'+label,h)
                    put('write_error_'+label,(estimate-write).square().mean(-1))
                    s['a_predictions'][label]=down(estimate)
                s['a_predictions']['full']=down(write)
                s['a_predictions']['zero']=down(t.zeros_like(write))

            def record_a(m,args):
                a=args[0][:,-1]
                if split=='discovery':sensitivity[:]+=(a.float()-s['zero']).square().sum(0).cpu().numpy()
                elif split=='confirm':
                    current=a[:,units].float()
                    put('actual_a',current)
                    for label,value in s['a_predictions'].items():
                        put('a_error_'+label,(current-value).square().mean(-1))
                        if label.endswith('selected_128'):put('a_'+label,value)

            def residual(m,args):s['res']=args[0][:,-1].clone()
            layer.register_forward_pre_hook(residual)
            att.register_forward_pre_hook(prepare,with_kwargs=True)
            att.q_proj.register_forward_hook(capture('q',QS))
            att.k_proj.register_forward_hook(capture('k',KS))
            att.v_proj.register_forward_hook(capture('v',KS))
            att.o_proj.register_forward_pre_hook(before_o)
            layer.mlp.down_proj.register_forward_pre_hook(record_a)
            return layer

        def callback(b,layer,x,y,batch,width,offset=0,**kw):
            if b==BLOCK-1:
                ep=[width-len(r['tokens'])+r['endpoint'] for r in batch]
                part=y[t.arange(len(batch),device='cuda'),ep]
                pre_prefix.append(int((part[::2]==part[1::2]).all(-1).sum()))
            if b>=0 and offset+len(batch)==len(rr) and (b%5==4 or b==BLOCK):progress(shard,b)

        rt.block=block
        with GpuLease(),t.inference_mode():
            _,logits=rt.forward(rr,callback)
            scores=rt.score(logits,rr)
            del logits
        data={k:np.concatenate(v) for k,v in saved.items()}
        if split=='discovery':data.update(mom=mom,counts=counts,sensitivity=sensitivity)
        archive(name+'.npz',data)
        error=float(data['sdpa'].max()); equal=sum(pre_prefix)
        commit(name+'.json',dict(sha=sha(OUT/(name+'.npz')),scores=scores,max_sdpa=error,prefix_equal=equal,
                                prereg_sha=sha(OUT/'prereg.json'),seal_sha=sha(OUT/'candidate.json') if split=='confirm' else None))
        assert error<=CAP and equal==len(rr)//2,(name,error,equal)
        pairs+=equal; technical.append(error); all_scores.extend(scores); shards[name+'.npz']=sha(OUT/(name+'.npz'))
    cells,passed=behavior(rows,all_scores)
    commit(f'{split}-collection.json',dict(provenance=pre['provenance'],passed=passed,cells=cells,shards=shards,prefix_equal=pairs,max_sdpa=max(technical),elapsed=time.monotonic()-started,torch=t.__version__,transformers=transformers.__version__))
    print(f'{split} complete, behavior gate={passed}, strict={sum(c[0] for c in cells)}, top1={sum(c[1] for c in cells)}',flush=True)


def consume(split):
    c=read(OUT/f'{split}-collection.json')
    data=[]
    for name,expected in sorted(c['shards'].items()):
        assert sha(OUT/name)==expected
        with np.load(OUT/name) as z:data.append({k:z[k] for k in z.files})
    return c,data


def fit_model():
    from source_binding_runtime import Runtime
    pre=checked()
    c,data=consume('discovery')
    assert c['passed']
    mom=np.sum([x['mom'] for x in data],0)
    n=np.sum([x['counts'] for x in data],0)
    mean=mom[:,0]/n[:,None]
    variance=np.maximum(mom[:,1]/n[:,None]-mean**2,0)
    scores=np.sum([x['sensitivity'] for x in data],0)
    units=np.argsort(-scores,kind='stable')[:32]
    rt=Runtime('qwen14'); prefix=f'model.layers.{BLOCK}.'
    rng=np.random.default_rng(SEED+17)
    result=dict(units=units,sensitivity=scores)
    for k in BRANCHES:
        w=rt.weight(prefix+f'self_attn.{k}_proj.weight','cpu')[QS if k=='q' else KS].contiguous()
        j=int(k!='q')
        rank=variance[j]*np.square(w.float().numpy().astype(np.float64)).sum(0)
        idx=np.argsort(-rank,kind='stable'); rand=rng.permutation(rt.dim)
        result.update({f'mean_{k}':mean[j].astype(np.float32),f'rank_{k}':rank,f'weight_{k}':w.view(rt.t.uint16).numpy()})
        for n in (32,128):result.update({f'{k}_selected_{n}':idx[:n],f'{k}_random_{n}':rand[:n]})
    for k,name,region,columns in [('o','self_attn.o_proj.weight',QS,True),('gate','mlp.gate_proj.weight',units,False),('up','mlp.up_proj.weight',units,False),('down','mlp.down_proj.weight',units,True)]:
        w=rt.weight(prefix+name,'cpu')
        result['weight_'+k]=(w[:,region] if columns else w[region]).contiguous().view(rt.t.uint16).numpy().copy()
    z=(np.concatenate([x['q_final'] for x in data]).astype(np.uint32)<<16).view(np.float32)[:,result['q_selected_32']].astype(np.float64)
    center=z.mean(0); scale=np.maximum(z.std(0),1e-6); z=(z-center)/scale
    y=np.tile(np.array([r[-1] for r in CELLS])-.5,SIZES['discovery'])
    coef=np.linalg.solve(z.T@z+np.eye(32),z.T@y)
    result.update(probe_center=center,probe_scale=scale,probe_coef=coef)
    archive('predictor.npz',result)
    commit('candidate.json',dict(utc=datetime.now(timezone.utc).isoformat(),prereg_sha=sha(OUT/'prereg.json'),predictor_sha=sha(OUT/'predictor.npz'),units=units.tolist(),provenance=pre['provenance']))
    print('Sealed branch coordinates, probe and MLP units',flush=True)


def evaluate():
    pre=checked(); c,data=consume('confirm'); rows=materials('confirm',pre); seal()
    a={k:np.concatenate([x[k] for x in data]) for k in data[0]}
    fit=np.load(OUT/'predictor.npz')
    idx=np.array([r['conflict'] for r in rows]); x=np.arange(len(rows)); names=np.array([r['name_slot'] for r in rows]); line=np.array([r['row_slot'] for r in rows]); gold=np.array([r['target_record'] for r in rows]); rule=np.array([r['rule'] for r in rows]); fam=np.array([r['family'] for r in rows])
    boot=np.random.default_rng(SEED+29).integers(0,SIZES['confirm'],(10000,SIZES['confirm']))
    def interval(value,which=idx):
        f=np.bincount(fam[which],weights=np.asarray(value)[which],minlength=SIZES['confirm'])/np.bincount(fam[which],minlength=SIZES['confirm'])
        return [float(f.mean()),*np.quantile(f[boot].mean(1),[.0125,.9875]).tolist()]
    def difference(mass):return np.log(np.maximum(mass[x,names],1e-20))-np.log(np.maximum(mass[x,line],1e-20))
    orig=a['mass_actual']; routing=orig.argmax(1); observed=difference(orig); priority={}
    for r in (0,1):
        flag=idx&(rule==r)
        priority[str(r)]=dict(name_top=float((routing[flag]==names[flag]).mean()),row_top=float((routing[flag]==line[flag]).mean()),gold_top=float((routing[flag]==gold[flag]).mean()),D=interval(observed,flag),among_pair_name=float((observed[flag]>0).mean()))
    crossover=interval(np.column_stack([(observed[::2]-observed[1::2])]*2).ravel(),idx & (rule==0))
    switches=bool(crossover[1]>0 and min(g['gold_top'] for g in priority.values())>=.75)
    measurements={}; details={}
    for k,m,n in ARMS:
        label=f'{k}_{m}_{n}'; mass=a['mass_'+label]
        if k=='v':assert np.array_equal(mass,orig)
        matched=(mass.argmax(1)==routing)
        errors={'D':np.square(difference(mass)-observed),'write':a['write_error_'+label],'A':a['a_error_'+label]}
        details[label]=errors
        measurements[label]=dict(route=float(matched[idx].mean()),**{u:float(v[idx].mean()) for u,v in errors.items()})
    gates={}
    for k in BRANCHES:
        selected=f'{k}_selected_128'; comparisons={}
        for base in (f'{k}_mean_0',f'{k}_random_128'):
            for field in (('D',) if k in ('q','k') else ('write','A')):
                diff=interval(details[base][field]-details[selected][field]); gain=1-measurements[selected][field]/max(measurements[base][field],1e-20)
                comparisons[f'{base}:{field}']=dict(gain=gain,paired=diff,passed=gain>=.20 and diff[1]>0)
        gates[k]=dict(passed=bool(c['passed'] and all(v['passed'] for v in comparisons.values()) and (k=='v' or measurements[selected]['route']>=.9)),comparisons=comparisons)
    qq=(a['q_final'].astype(np.uint32)<<16).view(np.float32)[:,fit['q_selected_32']]
    decode=((qq-fit['probe_center'])/fit['probe_scale'])@fit['probe_coef']>0
    result=dict(provenance=pre['provenance'],collection_sha=sha(OUT/'confirm-collection.json'),candidate_sha=sha(OUT/'candidate.json'),behavior_pass=c['passed'],families=SIZES['confirm'],
                conflict_inputs=int(idx.sum()),rule_diagnostics=priority,crossover=crossover,follows_rule=switches,probe_accuracy=float((decode==rule).mean()),
                branches=measurements,gates=gates,causal_admitted=[k for k,v in gates.items() if v['passed']],full_mlp_floor=float(a['a_error_full'][idx].mean()),zero_mlp=float(a['a_error_zero'][idx].mean()))
    commit('evaluation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='branches'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    command=sys.argv[1]
    if command in ('pilot','discovery','confirm'):collect(command)
    else:{'freeze':freeze,'select':fit_model,'evaluate':evaluate}[command]()
