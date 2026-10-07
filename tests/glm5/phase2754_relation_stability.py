"""Approximate relational regularities, paired lexical controls, native sources."""
import argparse,gc,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from phase2752_context_interaction import ROOT,write,sha,now

OUT=ROOT/'tests/glm5/result/rdc_relation_stability_20260923'
FAMILIES=('category','role','spatial','containment')
CUTS=(4,8,12,16)

def snapshot(path):
    dest=OUT/'code_snapshots'/(sha(path)+path.suffix);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.write_bytes(path.read_bytes())
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),snapshot=str(dest.relative_to(ROOT)))

def decode(bits):return (bits.astype(np.uint32)<<16).view(np.float32)

def render(fam,path,a,b,template,eta):
    passive=template in (1,3)
    if fam=='category':
        facts=[f'{y} has {x} as a strict subcategory.' if passive else f'{x} is a strict subcategory of {y}.' for x,y in zip(path[:-1],path[1:])]
        rule='Strict subcategory relations are transitive and never hold in both directions.'
        claim=f'{b} has {a} as a strict subcategory' if passive else f'{a} is a strict subcategory of {b}'
    elif fam=='spatial':
        facts=[f'{y} is to the right of {x}.' if passive else f'{x} is to the left of {y}.' for x,y in zip(path[:-1],path[1:])]
        rule='All objects lie on one line. Left and right are opposite directions.'
        claim=f'{b} is to the right of {a}' if passive else f'{a} is to the left of {b}'
    elif fam=='containment':
        facts=[f'{y} contains {x}.' if passive else f'{x} is inside {y}.' for x,y in zip(path[:-1],path[1:])]
        rule='These containers are strictly nested. No container is inside itself.'
        claim=f'{b} contains {a}' if passive else f'{a} is inside {b}'
    else:
        facts=[f'Event {k+1}: '+(f'The key was handed to {y} by {x}.' if passive else f'{x} handed the key to {y}.') for k,(x,y) in enumerate(zip(path[:-1],path[1:]))]
        rule='Events happened in numerical order. All people named here are different.'
        claim=f'{b} received the key last, and {a} gave it first' if passive else f'the transfer chain started with {a} and ended with {b}'
    facts=[f'Fact {i+1}: '+f if fam!='role' else f for i,f in enumerate(facts)]
    if template==2:context='Use this fact list:\n'+'\n'.join('- '+f for f in facts)+'\n'+rule
    elif template==3:context='Read this record: '+' / '.join(facts)+' '+rule
    else:context='Use only these facts. '+' '.join(facts)+' '+rule
    prefix=context+'\nStatement: It is '+('really' if eta==1 else 'not')+' the case that '
    text=prefix+claim+'.\nReturn yes if and only if the entire statement is true; otherwise return no.\nAnswer:'
    spans={name:[len(prefix)+claim.index(name),len(prefix)+claim.index(name)+len(name)] for name in (a,b)}
    return text,spans

def prepare():
    if (OUT/'material.json').exists():return json.loads((OUT/'material.json').read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    toks={s:AutoTokenizer.from_pretrained(ROOT/'models/hf'/m,local_files_only=True) for s,m in [('4B','qwen3-4b'),('14B','Qwen3-14B')]}
    syll=('ba','ce','di','fo','gu','ha','ji','ko','lu','me','ni','po','ra','se','ti','vu')
    pool=[(''.join(syll[(i//16**k)%16] for k in (3,2,1,0))+'z').capitalize() for i in range(65536)]
    np.random.default_rng(2754001).shuffle(pool)
    worlds=[];rows=[]
    for fam in FAMILIES:
        for split,count in [('train',24),('validation',8),('entity',16),('surface',16),('depth',16)]:
            for j in range(count):
                wid=f'{fam}_{split}_{j:02d}';names=pool[len(worlds)*4:len(worlds)*4+4];depth=3 if split=='depth' else 1+j%2
                worlds.append(dict(id=wid,family=fam,split=split,index=j,names=names,depth=depth))
                for template in ((2,3) if split=='surface' else (0,1)):
                    group=f'{wid}_t{template}'
                    for theta in (-1,1):
                        path=names[:depth+1] if theta==1 else names[:depth+1][::-1]
                        for rho in (-1,1):
                            a,b=(names[0],names[depth]) if rho==1 else (names[depth],names[0])
                            for eta in (-1,1):
                                # Independent graph reachability check, before model scoring.
                                edges=list(zip(path[:-1],path[1:]));reach={a}
                                for _ in path:reach.update(y for x,y in edges if x in reach)
                                predicate=(b in reach) if fam!='role' else (a==path[0] and b==path[-1])
                                assert (1 if predicate else -1)==theta*rho
                                text,spans=render(fam,path,a,b,template,eta);tm={}
                                for side,tok in toks.items():
                                    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
                                    slots=[]
                                    for name in (a,b):
                                        lo,hi=spans[name];ii=[k for k,(u,v) in enumerate(enc['offset_mapping']) if u<hi and v>lo];assert ii
                                        slots.append(ii[-1])
                                    tm[side]=dict(token_ids=enc['input_ids'],length=len(enc['input_ids']),query_source_target_last_tokens=slots)
                                cell=(int(theta==1)*4+int(rho==1)*2+int(eta==1))
                                rows.append(dict(id=f'{group}_c{cell}',group=group,world=wid,family=fam,split=split,index=j,template=template,depth=depth,
                                    theta=theta,rho=rho,eta=eta,cell=cell,predicate_sign=theta*rho,truth_sign=theta*rho*eta,
                                    expected='yes' if theta*rho*eta==1 else 'no',text=text,edges=edges,query=[a,b],tokenization=tm,
                                    replicate14B=split in ('entity','surface','depth') and j<2 and template==((2 if split=='surface' else 0)+j%2)))
    assert len(worlds)==320 and len(rows)==5120
    assert sum(r['replicate14B'] for r in rows)==192
    groups={}
    for r in rows:groups.setdefault(r['group'],[]).append(r)
    bag_checks={}
    for side in toks:
        for gg in groups.values():
            assert len({r['tokenization'][side]['length'] for r in gg})==1
            for eta in (-1,1):
                rr=[r for r in gg if r['eta']==eta];ref=Counter(rr[0]['tokenization'][side]['token_ids'])
                assert all(Counter(r['tokenization'][side]['token_ids'])==ref for r in rr),('token bag mismatch',side,rr[0]['id'])
        bag_checks[side]=True
    mat=dict(created_utc=now(),worlds=worlds,rows=rows,sampling_unit='world;2 templates x8 factorial conditions are repeated measures')
    write(OUT/'material.json',mat)
    write(OUT/'design.json',dict(created_utc=now(),phase=2754,question='Stable approximate relation/role/negation coding despite imperfect model behavior.',
        factors=dict(theta='Fact chain orientation',rho='Query semantic source/target binding',eta='Affirmed+1 or negated-1 proposition'),
        exact_labels='Predicate theta*rho; statement truth theta*rho*eta. Eight cells balanced, token bags equal at fixed eta; independently checked via graph reachability.',
        discovery='4B train96worlds and validation32worlds only; no old outcomes fit. Freeze probe choice before confirmation192worlds.',
        replication14B='192 matched prompts,24worlds,one template/world; same raw protocol, sequential loading. Limited paired checkpoint comparison, not pure parameter-count causal test.',
        response='Canonical yes-minus-no logit, aggregate yes/no mass, first-token correctness, paired true-v-false ranking, factorial truth coefficient. No perfection gate or dropping erroneous worlds.',
        hypotheses=['Truth-aligned relational effects can persist while global yes/no bias causes errors.','Relation effects may be more stable across templates than individual outputs.','Actual attention/MLP residual writes can be decomposed along fixed output weights, including numerical rounding.'],
        candidate_probes='Train-only full-coordinate ridge: end state versus end+two semantic query-slot states at4,8,12,16; static length baseline. Separate labels predicate,statement truth,and actual model margin. Finalnorm probe only descriptive decodability ceiling.',
        alpha=[.001,.01,.1,1,10],selection='Validation balanced truth accuracy for semantic probe; validation MSE for model-margin forecast. No refit or test selection.',
        inference='World bootstrap2000 draws, stratified family. Primary approximate stability evaluated across predefined axes/families, not universal perfection.',
        capture='All native last-position states and attention/MLP writes; last-token query-slot states at4/8/12/16. Lossless BF16 bits in uint16. No all-token or full-KV claim.',
        bounds='4B discovery<=600s,confirmation<=900s;14Bpilot<=180s,formal<=1800s; offline analyses<=600s per process. No concurrent GPU model loads.',
        no_claims=['Probe decoding is not proof the model uses that decoder.','Factorial decomposition is a definition, not an automatically discovered mechanism.','Small-model roughness is a tested limitation, not an unlimited failure exemption.']))
    write(OUT/'material_seal.json',dict(created_utc=now(),source=snapshot(Path(__file__)),material_sha256=sha(OUT/'material.json'),design_sha256=sha(OUT/'design.json'),token_bag_checks=bag_checks))
    print(dict(worlds=320,prompts=5120,matched14B=192,bag_checks=bag_checks),flush=True)
    return mat

def parameter_vectors(model_dir,yes,no):
    from safetensors import safe_open
    cfg=json.loads((model_dir/'config.json').read_text());index=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    key='model.embed_tokens.weight' if cfg.get('tie_word_embeddings') else 'lm_head.weight'
    with safe_open(model_dir/index[key],framework='pt',device='cpu') as f:
        ws=f.get_slice(key);direction=(ws[yes:yes+1].float()-ws[no:no+1].float())[0].numpy()
    with safe_open(model_dir/index['model.norm.weight'],framework='pt',device='cpu') as f:gamma=f.get_tensor('model.norm.weight').float().numpy()
    return direction,gamma

def collect(side,stage):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    mat=prepare()
    if stage=='confirmation':assert (OUT/'selection.json').exists(),'Select on discovery before confirmation.'
    rows=[r for r in mat['rows'] if (r['split'] in ('train','validation') if stage=='discovery' else r['split'] not in ('train','validation'))]
    if side=='14B':rows=[r for r in rows if r['replicate14B']]
    if stage=='pilot':rows=rows[:8]
    dest=OUT/side/stage;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'done.json').exists():print('Completed capture reused',dest,flush=True);return
    assert not list(dest.glob('chunk_*.npz')),'Preserve partial captures.'
    budget=(180 if side=='14B' else 180) if stage=='pilot' else (1800 if side=='14B' else 600 if stage=='discovery' else 900)
    import threading,os
    def stop():write(dest/'budget_exceeded.json',dict(created_utc=now(),seconds=budget));os._exit(124)
    watchdog=threading.Timer(budget,stop);watchdog.start()
    start=time.time();torch.set_num_threads(8);torch.manual_seed(2754001);torch.backends.cuda.matmul.allow_tf32=False
    mdir=ROOT/'models/hf'/('qwen3-4b' if side=='4B' else 'Qwen3-14B')
    tok=AutoTokenizer.from_pretrained(mdir,local_files_only=True)
    yes=tok.encode(' yes',add_special_tokens=False);no=tok.encode(' no',add_special_tokens=False);assert len(yes)==len(no)==1
    direction,gamma=parameter_vectors(mdir,yes[0],no[0]);v=(direction*gamma).astype(np.float64)
    np.savez(dest/'parameter_readout.npz',direction=direction,gamma=gamma)
    yesno={a:sorted({tok.encode(s,add_special_tokens=False)[0] for s in (a,a.capitalize(),' '+a,' '+a.capitalize()) if len(tok.encode(s,add_special_tokens=False))==1}) for a in ('yes','no')}
    execution=dict(created_utc=now(),source=snapshot(Path(__file__)),material_sha256=sha(OUT/'material.json'),model_config_sha256=sha(mdir/'config.json'),
        selected_ids=[r['id'] for r in rows],torch=torch.__version__,precision='BF16 native, uint16 exact bit storage',batch=1,cache=False,attention='eager',
        padding=False,truncation=False,canonical_tokens=dict(yes=yes[0],no=no[0]),yesno_token_sets=yesno,watchdog_seconds=budget)
    write(dest/'execution.json',execution)
    opts=dict(dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    if side=='14B':opts.update(device_map='auto',max_memory={0:'10GiB','cpu':'48GiB'})
    model=AutoModelForCausalLM.from_pretrained(mdir,**opts).eval()
    if side=='4B':model.to('cuda')
    execution['load_seconds']=time.time()-start;execution['device_map']={k:str(vv) for k,vv in getattr(model,'hf_device_map',{'':'cuda'}).items()};write(dest/'execution.json',execution)
    def bits(t):return t.detach().contiguous().view(torch.uint16).cpu().numpy().copy()
    store={};hooks=[]
    def hook(key,pre=False):
        def rec(module,args,output=None):
            value=args[0] if pre else output
            if isinstance(value,tuple):value=value[0]
            store[key]=bits(value[0,-1])
        return rec
    for i,layer in enumerate(model.model.layers):
        hooks.extend([layer.self_attn.register_forward_hook(hook(('a',i))),layer.mlp.register_forward_hook(hook(('m',i)))])
    hooks.append(model.model.norm.register_forward_pre_hook(hook('raw',True)))
    banks={k:[] for k in ('hidden','attention','mlp','query_slots')};metadata=[];chunk=0;maxerr=0.
    def flush():
        nonlocal metadata,chunk
        if not metadata:return
        np.savez(dest/f'chunk_{chunk:03d}.npz',**{k:np.stack(vv) for k,vv in banks.items()})
        write(dest/f'chunk_{chunk:03d}.json',metadata)
        for vv in banks.values():vv.clear()
        metadata=[];chunk+=1
    try:
        for idx,row in enumerate(rows):
            store.clear();tm=row['tokenization'][side]
            ids=torch.tensor([tm['token_ids']],device=model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                result=model(input_ids=ids,use_cache=False,output_hidden_states=True,logits_to_keep=1)
                hidden=np.stack([bits(x[0,-1]) for x in result.hidden_states]+[store['raw']])
                attn=np.stack([store[('a',i)] for i in range(len(model.model.layers))]);mlp=np.stack([store[('m',i)] for i in range(len(model.model.layers))])
                query=np.stack([bits(result.hidden_states[c][0,tm['query_source_target_last_tokens']]) for c in CUTS])
                hh=decode(hidden);aa=decode(attn);mm=decode(mlp)
                # Check every layer's BF16 residual equation using native rounding.
                before=hh[:len(model.model.layers)];after=np.concatenate([hh[1:len(model.model.layers)],hh[-1:]])
                reconstruct=(torch.from_numpy(before).to(torch.bfloat16)+torch.from_numpy(aa).to(torch.bfloat16))+torch.from_numpy(mm).to(torch.bfloat16)
                error=float(np.max(abs(reconstruct.float().numpy()-after)));maxerr=max(maxerr,error);assert error==0
                before_num=before.astype(np.float64)@v;after_num=after.astype(np.float64)@v
                a_num=aa.astype(np.float64)@v;m_num=mm.astype(np.float64)@v
                rounding=after_num-before_num-a_num-m_num
                raw=hh[-1].astype(np.float64);rms=float(np.sqrt(np.mean(raw**2)+model.config.rms_norm_eps))
                numerator=float(raw@v);fp32_margin=float(hh[-2].astype(np.float64)@direction)
                logits=result.logits[0,-1].float();lp=logits.log_softmax(-1);p=int(logits.argmax())
                vals,inds=torch.topk(logits,20)
                metadata.append(dict(id=row['id'],group=row['group'],world=row['world'],cell=row['cell'],split=row['split'],
                    prediction_id=p,prediction_text=tok.decode([p]),margin=float(logits[yes[0]]-logits[no[0]]),fp32_margin=fp32_margin,
                    yesno_logmass={a:float(torch.logsumexp(lp[ts],0)) for a,ts in yesno.items()},
                    numerator=numerator,rms=rms,norm_rounding_margin=fp32_margin-numerator/rms,
                    source_numerator=dict(initial=float(before_num[0]),attention=a_num.tolist(),mlp=m_num.tolist(),rounding=rounding.tolist()),
                    residual_anchor_max=error,top20_ids=inds.cpu().tolist(),top20_logits=vals.cpu().tolist(),logsumexp=float(logits.logsumexp(0))))
                for key,value in [('hidden',hidden),('attention',attn),('mlp',mlp),('query_slots',query)]:banks[key].append(value)
                del result,logits,lp
            if (idx+1)%32==0:flush();print(f'{side} {stage} {idx+1}/{len(rows)} {time.time()-start:.1f}s',flush=True)
        flush()
        write(dest/'done.json',dict(created_utc=now(),count=len(rows),worlds=len({r['world'] for r in rows}),groups=len({r['group'] for r in rows}),elapsed_seconds=time.time()-start,
            layers=len(model.model.layers),dim=model.config.hidden_size,peak_cuda_bytes=torch.cuda.max_memory_allocated(),residual_anchor_max=maxerr,
            boundary='0 embedding;1..L-1 residual;L finalnorm;L+1 raw final residual',query_cuts=CUTS,
            source_note='Attention/MLP writes in actual forward, projected onto fixed final output direction with gamma; intermediate sums are not counterfactual outputs.',source_sha256=sha(Path(__file__))))
    finally:
        watchdog.cancel()
        for hook_handle in hooks:hook_handle.remove()
        del model,store;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect']);p.add_argument('--model',choices=['4B','14B'],default='4B');p.add_argument('--stage',choices=['pilot','discovery','confirmation'],default='discovery');a=p.parse_args()
    prepare() if a.mode=='prepare' else collect(a.model,a.stage)
