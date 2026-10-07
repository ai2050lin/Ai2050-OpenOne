"""Versioned measurement repair + grouped composition test. Never edits legacy runs.

Preparation is frozen before model loading. CUDA arms run one process at a time.
All model outputs are observations, not truth labels for the predictor. Composition
predicts the doubly conditioned state from three singly/base conditioned states;
it is not a text-only semantic mechanism extractor.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from rdc_trusted_measurements import silu_jvp,rms,rms_jvp,measure,energy,self_test,rankcorr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/glm5/result/rdc_trusted_rebuild_20260923'
OLD=ROOT/'tests/glm5/result/rdc_query_construction_20260913'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def literal(name,file):
    for node in ast.parse(file.read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'material.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    fam=literal('FAM',ROOT/'tests/glm5/phase3100_omega_p98_upstream_predict.py')
    for f,v in fam.items():
        for i,body in enumerate(v['bodies']):
            for c,prefix in enumerate(v['prefixes']):
                rows.append(dict(id=f'legacy_{f}_{i}_{c}',group=f'legacy_{f}_{i}',family=f,split='legacy',
                                 cond=c,text=(prefix+' '+body) if prefix else body,expected=None))
    # Unique entities shared deliberately across two wording templates. All variants
    # of a source proposition stay in the same group; the grouping is explicit.
    names=['Arin','Bela','Ciro','Dena','Eron','Fara','Galen','Hana',
           'Iven','Jora','Kelan','Luma','Miro','Nera','Orin','Pela']
    objects=['copper key','violet box','silver cup','green scarf','wooden ring','blue stone','paper crown','orange shell',
             'brass coin','velvet pouch','glass bead','linen ribbon','marble disk','yellow feather','iron bell','red tile']
    for f in ('category','role','chain'):
        for i,name in enumerate(names):
            other=names[(i+5)%16]; obj=objects[i]
            for t in (0,1):
                split=('development' if i<8 else 'entity_holdout') if t==0 else ('wording_holdout' if i<8 else 'joint_holdout')
                gid=f'{f}_{i}_t{t}'
                for style in (0,1):
                    for neg in (0,1):
                        # Balanced truth at both negation levels, avoiding a yes-only baseline.
                        relation_true=i%2==0
                        if f=='category':
                            label='dax' if relation_true else 'wug'
                            facts=f'{name} is a dax. Every dax is a pel. No dax is a wug.'
                            query=f'Is {name} '+('not ' if neg else '')+f'a {label}?'
                            if t: facts=f'All daxes are pels and no dax is a wug. {name} belongs to the dax category.'
                        elif f=='role':
                            chosen=name if relation_true else other
                            facts=f'{name} gave the {obj} to {other}. The giver and recipient are different people.'
                            query=f'Was {chosen} '+('not ' if neg else '')+'the giver?'
                            if t:facts=f'The {obj} was given to {other} by {name}. They are different people.'
                        else:
                            chosen='left' if relation_true else 'right'
                            facts=f'{name} is to the left of {other}. {other} is to the left of the tower. Left and right are opposites.'
                            query=f'Is {name} '+('not ' if neg else '')+f'to the {chosen} of the tower?'
                            if t:facts=f'The tower is to the right of {other}, who is to the right of {name}. Left and right are opposites.'
                        lead='Use a formal tone. ' if style else ''
                        qintro='Question: ' if t==0 else 'Based only on these statements, answer this question: '
                        text=lead+facts+'\n'+qintro+query+'\nAnswer yes or no.\nAnswer:'
                        rows.append(dict(id=f'{gid}_s{style}n{neg}',group=gid,source_family=f'{f}_{i}',family=f,
                            split=split,cond=2*style+neg,style=style,negation=neg,wording=t,entity_index=i,
                            text=text,expected='yes' if relation_true != bool(neg) else 'no'))
    assert len(rows)==480 and len(set(r['id'] for r in rows))==480
    material=dict(created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
                  independent_source_groups=48,wording_groups=96,
                  note='New corpus has 48 semantic source groups, 2 wordings, 4 conditions = 384 prompts. Not 384 independent observations. Legacy 96 prompts are repair only.')
    write(path,material)
    write(OUT/'preregistered_design.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),material_sha256=sha(path),
        objective='Repair trustworthy local measurements; assess conditional state composition on grouped unseen entities/wordings.',
        inference='Descriptive within-model grouped bootstrap; no universal or cross-model p claim.',
        primary=['SwiGLU correct vs old relative error and cosine','RMSNorm JVP error','energy cross term and signed projected contributions',
                 'pre-o_proj head correspondence on identical legacy pairs','heldout h11 prediction from h10+h01-h00 versus h00, h10, h01',
                 'development-only mean interaction correction, per family, evaluated unchanged on holdouts'],
        scope='last-token all native coordinates of all hidden-state outputs; full last-block operands and logits. Not all-token field.',
        leakage='No h11 holdout or answer label enters predictors. Three source states are assumed measured; not a text-only predictor.',
        allocation='4B smoke 12 prompts, then 480 full if reliable. 14B sequential pilot before matching run; CPU offload if required.',
        budget='Bounded pilot first; per-arm collector cap 1800 seconds; preserve partial captures on timeout; no concurrent model load.',
        arithmetic='native bf16 eager, batch1, no cache, no padding/truncation; local reconstruction/JVP torch.float64 on CUDA.',
        planned_uncertainty='2000 bootstrap draws over source group, not token/condition; development and each holdout reported separately.',
        model_config_sha256={s:sha(ROOT/'models/hf'/m/'config.json') for s,m in [('4B','qwen3-4b'),('14B','Qwen3-14B')]},
        math_checks=self_test()))
    return material


def collect(side,smoke=False,limit=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    material=prepare(); rows=material['rows']
    if smoke:rows=rows[:8]+rows[96:100]
    elif side=='14B':
        # Cost-only decision frozen after the pilot, before any full 14B outcomes.
        # Balance both entity halves, both wordings and all three families.
        rows=[r for r in rows if r['split']=='legacy' or r['entity_index']%8<4]
        amendment=OUT/'14B_resource_amendment.json'
        if not amendment.exists():
            write(amendment,dict(created_utc=datetime.now(timezone.utc).isoformat(),
                reason='12-prompt offload pilot 65.57s; 480 may exceed 1800s cap. Select balanced IDs by cost, not outcomes.',
                n_prompts=288,n_new_prompts=192,n_new_semantic_sources=24,
                selection='legacy all96; new entity_index modulo8 <4; both wordings and all4 conditions',
                selected_ids=[r['id'] for r in rows],pilot='14B_smoke/capture_done.json',
                analysis='14B uncertainty uses 12 groups per split, not24; compare matched subset when comparing models.'))
    if limit:rows=rows[:limit]
    dest=OUT/(side+('_smoke' if smoke else ''))
    dest.mkdir(exist_ok=True)
    if (dest/'capture_done.json').exists():
        print('Capture already complete; reuse',dest,flush=True);return
    assert not any(dest.glob('chunk_*.npz')), 'Partial existing run: preserve and use resume, do not overwrite.'
    mdir=ROOT/'models/hf'/('qwen3-4b' if side=='4B' else 'Qwen3-14B')
    torch.set_num_threads(8);torch.manual_seed(2751001)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.time()
    config=dict(side=side,smoke=smoke,n_requested=len(rows),start_utc=datetime.now(timezone.utc).isoformat(),
                torch=torch.__version__,script_sha256=sha(Path(__file__)),math_sha256=sha(Path(__file__).with_name('rdc_trusted_measurements.py')),
                material_sha256=sha(OUT/'material.json'),dtype='bfloat16',device_policy='cuda' if side=='4B' else 'auto max CUDA 10GiB CPU 48GiB')
    write(dest/'execution.json',config)
    tok=AutoTokenizer.from_pretrained(mdir,local_files_only=True)
    options=dict(torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    if side=='14B':options.update(device_map='auto',max_memory={0:'10GiB','cpu':'48GiB'})
    model=AutoModelForCausalLM.from_pretrained(mdir,**options).eval()
    if side=='4B':model=model.to('cuda')
    layers=model.model.layers;last=layers[-1]
    config['device_map']={k:str(v) for k,v in getattr(model,'hf_device_map',{'':'cuda'}).items()}
    config['load_seconds']=time.time()-start
    config['model_parameters']=sum(p.numel() for p in model.parameters())
    write(dest/'execution.json',config)
    store={}
    def rec(tag,pre=False):
        def hook(mod,args,output=None):
            val=args[0] if pre else output
            if isinstance(val,tuple):val=val[0]
            store[tag]=val.detach()
        return hook
    handles=[last.register_forward_pre_hook(rec('h0',True)),last.self_attn.register_forward_hook(rec('a')),
             last.register_forward_hook(rec('h2')),last.mlp.register_forward_hook(rec('m')),
             last.mlp.register_forward_pre_hook(rec('x',True)),last.mlp.gate_proj.register_forward_hook(rec('g')),
             last.mlp.up_proj.register_forward_hook(rec('u')),last.mlp.down_proj.register_forward_pre_hook(rec('act',True)),
             model.model.norm.register_forward_pre_hook(rec('prenorm',True))]
    head_layer=37 if side=='14B' else len(layers)-2
    handles.extend([layers[head_layer].self_attn.o_proj.register_forward_pre_hook(rec('heads',True)),
                    layers[head_layer].self_attn.register_forward_hook(rec('head_out'))])
    banks={}; meta=[];anchors=[];chunk=0
    def flush():
        nonlocal chunk,banks,meta
        if not meta:return
        np.savez(dest/f'chunk_{chunk:03d}.npz',**{k:np.stack(v) for k,v in banks.items()})
        write(dest/f'chunk_{chunk:03d}.json',meta)
        chunk+=1;banks={};meta=[]
    try:
        for i,row in enumerate(rows):
            ids=tok(row['text'],add_special_tokens=False,return_tensors='pt')['input_ids']
            ids=ids.to(model.get_input_embeddings().weight.device)
            store.clear()
            with torch.inference_mode():
                result=model(input_ids=ids,use_cache=False,output_hidden_states=True)
                replay=last.mlp.down_proj(last.mlp.act_fn(store['g'])*store['u'])
                err=float((replay-store['m']).abs().max())
                chain=float(((store['h0']+store['a'])+store['m']-store['h2']).abs().max())
                anchors.append(dict(id=row['id'],mlp_replay=err,block_chain=chain))
                if err!=0 or chain!=0:raise RuntimeError(f'Anchor failed: {anchors[-1]}')
                for k,v in store.items():banks.setdefault(k,[]).append(v[0,-1].float().cpu().numpy())
                # Last item is post-final-norm; raw final residual is separately h2/prenorm.
                banks.setdefault('hidden',[]).append(np.stack([h[0,-1].float().cpu().numpy() for h in result.hidden_states]))
                banks.setdefault('logits',[]).append(result.logits[0,-1].float().cpu().numpy())
                pred=int(result.logits[0,-1].argmax())
                meta.append(dict(**row,token_ids=ids[0].cpu().tolist(),prediction_id=pred,prediction_text=tok.decode([pred]),
                                 hidden_boundary='embedding then intermediate residuals; last returned item post-final-norm'))
                del result,replay
            if (i+1)%16==0:
                flush();print(f'{side} captured {i+1}/{len(rows)} elapsed={time.time()-start:.1f}s',flush=True)
            if time.time()-start>1800:
                flush();write(dest/'partial.json',dict(captured=i+1,reason='1800s cap',elapsed=time.time()-start));return
        flush()
        # Persist weight slices required for independent reanalysis, from exact loaded
        # bf16 values. Not an extracted mechanism; they are the verification reference.
        write(dest/'anchors.json',anchors)
        # Accelerate may replace CPU-offloaded parameters by meta placeholders
        # outside forward. Read those exact checkpoint tensors, never a meta tensor.
        names=dict(Wg=f'model.layers.{len(layers)-1}.mlp.gate_proj.weight',
                   Wu=f'model.layers.{len(layers)-1}.mlp.up_proj.weight',
                   Wd=f'model.layers.{len(layers)-1}.mlp.down_proj.weight',
                   gamma=f'model.layers.{len(layers)-1}.post_attention_layernorm.weight',final_gamma='model.norm.weight')
        weights={}
        index=json.loads((mdir/'model.safetensors.index.json').read_text())['weight_map'] if (mdir/'model.safetensors.index.json').exists() else None
        from safetensors import safe_open
        for key,name in names.items():
            param=model.get_parameter(name)
            if param.device.type=='meta':
                file=mdir/index[name] if index else mdir/'model.safetensors'
                with safe_open(file,framework='pt',device='cpu') as sf:
                    weights[key]=sf.get_tensor(name).to(torch.bfloat16).float().numpy()
            else:weights[key]=param.detach().float().cpu().numpy()
        np.savez(dest/'lastblock_weights.npz',**weights)
        write(dest/'anchors.json',anchors)
        write(dest/'capture_done.json',dict(n=len(rows),elapsed=time.time()-start,chunks=chunk,head_layer=head_layer,
            model_config=model.config.to_dict(),cuda_peak_allocated=torch.cuda.max_memory_allocated(),
            precision='bf16 native; saved fp32 exactly embeds bf16 observations',partial=False,
            checkpoints={p.name:sha(p) for p in sorted(mdir.glob('*.safetensors'))} if smoke else 'same local checkpoint; smoke hashes if available'))
    finally:
        for h in handles:h.remove()
        del model,store
        gc.collect();torch.cuda.empty_cache()


def load_bank(dest):
    data={};rows=[]
    for p in sorted(dest.glob('chunk_*.npz')):
        with np.load(p,allow_pickle=False) as z:
            for k in z.files:data.setdefault(k,[]).append(z[k])
        rows.extend(json.loads(p.with_suffix('.json').read_text(encoding='utf-8')))
    return {k:np.concatenate(v) for k,v in data.items()},rows


def summarize(values):
    vals=np.array([v for v in values if v is not None],float)
    return dict(n=len(vals),median=float(np.median(vals)),q10=float(np.quantile(vals,.1)),q90=float(np.quantile(vals,.9))) if len(vals) else dict(n=0)


def analyze(side,smoke=False):
    dest=OUT/(side+('_smoke' if smoke else ''))
    done=json.loads((dest/'capture_done.json').read_text(encoding='utf-8'))
    data,rows=load_bank(dest);torch.set_num_threads(8)
    device='cuda';dtype=torch.float64
    with np.load(dest/'lastblock_weights.npz') as z:w={k:torch.tensor(z[k],dtype=dtype,device=device) for k in z.files}
    eps=done['model_config']['rms_norm_eps']
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['group'],{})[r['cond']]=i
    t=lambda key,i:torch.tensor(data[key][i],dtype=dtype,device=device)
    repairs=[];jvp_validation=[];legacy_order={fa:[] for fa in 'ABC'}
    for group,idx in groups.items():
        if 0 not in idx:continue
        b=idx[0];g,u=t('g',b),t('u',b)
        for cond,c in idx.items():
            if cond==0:continue
            dg,du=t('g',c)-g,t('u',c)-u
            truth=t('m',c)-t('m',b)
            correct=w['Wd']@silu_jvp(g,u,dg,du)
            s=torch.sigmoid(g);wrong=w['Wd']@((s+g*s*(1-s))*(u*dg+g*du))
            h=t('h0',b)+t('a',b)
            channels=[]
            for key in ('a','h0'):
                dx=rms_jvp(h,w['gamma'],t(key,c)-t(key,b),eps)
                channels.append(w['Wd']@silu_jvp(g,u,w['Wg']@dx,w['Wu']@dx))
            total=channels[0]+channels[1]
            delta_h=t('h0',c)+t('a',c)-h
            # Full fp64 local function, including both matrices and normalization:
            # finite-difference checks validate formulas at a true model working point.
            if len(jvp_validation)<4:
                def local(x):
                    xx=rms(x,w['gamma'],eps)
                    return w['Wd']@(torch.nn.functional.silu(w['Wg']@xx)*(w['Wu']@xx))
                xx=rms(h,w['gamma'],eps);gg=w['Wg']@xx;uu=w['Wu']@xx
                dd=rms_jvp(h,w['gamma'],delta_h,eps)
                j=w['Wd']@silu_jvp(gg,uu,w['Wg']@dd,w['Wu']@dd)
                fd=(local(h+1e-4*delta_h)-local(h-1e-4*delta_h))/(2e-4)
                validation=measure(j,fd)
                assert validation['relative_error']<1e-5, validation
                jvp_validation.append(dict(id=rows[c]['id'],finite_difference=validation))
            # Actual first-order final norm bridge, compared with exact fp64 norm difference.
            raw=t('prenorm',b);draw=t('prenorm',c)-raw
            norm_true=rms(raw+draw,w['final_gamma'],eps)-rms(raw,w['final_gamma'],eps)
            norm_pred=rms_jvp(raw,w['final_gamma'],draw,eps)
            record=dict(id=rows[c]['id'],group=group,split=rows[c]['split'],family=rows[c]['family'],cond=cond,
                corrected=measure(correct,truth),legacy_wrong=measure(wrong,truth),
                channel_prediction=measure(total,truth),channel_energy=energy(*channels),
                channel_vs_direct=measure(total,correct),final_norm_jvp=measure(norm_pred,norm_true))
            if rows[c]['split']=='legacy':
                hd=done['model_config']['head_dim'];dh=(data['heads'][c]-data['heads'][b]).reshape(-1,hd)
                record['head_norms']=np.linalg.norm(dh.astype(np.float64),axis=1).tolist()
                legacy_order[rows[c]['family']].append(record)
            repairs.append(record)
    summary={}
    for split in sorted(set(r['split'] for r in repairs)):
        rr=[r for r in repairs if r['split']==split]
        summary[split]={key:{metric:summarize([r[key][metric] for r in rr]) for metric in ('cos','relative_error','norm_ratio')}
                        for key in ('corrected','legacy_wrong','channel_prediction','final_norm_jvp')}
        summary[split]['energy']={key:summarize([r['channel_energy'][key] for r in rr]) for key in ('norm_a','norm_h','cross','projected_a','projected_h')}
    correspondence={}
    if side=='14B' and not smoke:
        p=next((OLD/'phase3093').glob('*arbitration/*.npz'))
        with np.load(p) as z:
            for fa,rr in legacy_order.items():
                raw_norms=np.array([r['head_norms'] for r in rr])
                shares=raw_norms/np.maximum(raw_norms.sum(axis=1,keepdims=True),1e-30)
                norms=np.median(shares,axis=0);top=np.argsort(-norms)[:8]
                old=z['TOP8_'+fa];r1=z['R1_ALLNH_'+fa]
                correspondence[fa]=dict(top8_correct_preprojection=top.tolist(),top8_causal=old.tolist(),overlap=len(set(top)&set(old)),
                    rho_signed_causal=rankcorr(norms,r1),rho_abs_causal=rankcorr(norms,np.abs(r1)),
                    natural_measure='median of within-sample head L2 fractions; same aggregation as legacy, corrected basis',
                    chance_overlap_mean=8*8/len(norms))
    write(dest/'measurement_repair.json',dict(summary=summary,rows=repairs,working_point_finite_difference=jvp_validation,
        head_correspondence=correspondence,scope='Finite-difference local component checks are not proof of semantic mechanism.',
        math_checks=self_test(),analysis_script_sha256=sha(Path(__file__)),analysis_math_sha256=sha(Path(__file__).with_name('rdc_trusted_measurements.py'))))
    del w;gc.collect();torch.cuda.empty_cache()
    composition(dest,data,rows)
    print(json.dumps(dict(side=side,smoke=smoke,summary=summary),ensure_ascii=False),flush=True)


def composition(dest,data,rows):
    groups={}
    for i,r in enumerate(rows):
        if r['split']!='legacy':groups.setdefault(r['group'],{})[r['cond']]=i
    # Per-family correction is fit ONLY on development h11; every holdout group is unseen.
    means={}
    for family in ('category','role','chain'):
        ints=[]
        for group,idx in groups.items():
            if len(idx)!=4 or rows[idx[0]]['split']!='development' or rows[idx[0]]['family']!=family:continue
            a,b,c,d=[data['hidden'][idx[k]].astype(np.float64) for k in range(4)]
            ints.append(d-c-b+a)
        if ints:means[family]=np.mean(ints,axis=0)
    results=[]
    for group,idx in groups.items():
        if len(idx)!=4:continue
        a,b,c,d=[data['hidden'][idx[k]].astype(np.float64) for k in range(4)]
        target=d-a;den=np.linalg.norm(target,axis=1)
        estimates=dict(zero=np.zeros_like(target),negation=b-a,style=c-a,additive=b+c-2*a)
        family=rows[idx[0]]['family']
        if family in means:estimates['dev_mean_interaction']=estimates['additive']+means[family]
        errors={k:np.divide(np.linalg.norm(v-target,axis=1),den,out=np.full_like(den,np.nan),where=den>1e-12).tolist() for k,v in estimates.items()}
        # Direct first-token behavior, no teacher forcing. Longer natural completion remains untested.
        behavior=[]
        for k in range(4):
            r=rows[idx[k]];got=r['prediction_text'].strip().lower()
            behavior.append(dict(cond=k,expected=r['expected'],first_token=r['prediction_text'],exact_yes_no=got==r['expected']))
        results.append(dict(group=group,source_family=rows[idx[0]].get('source_family'),split=rows[idx[0]]['split'],family=family,
                            relative_errors=errors,first_token=behavior))
    # NaN only represents a degenerate unchanged source coordinate field (often embedding).
    for r in results:
        for key,val in r['relative_errors'].items():r['relative_errors'][key]=[v if np.isfinite(v) else None for v in val]
    write(dest/'composition.json',dict(groups=results,definition='Target double-condition change h11-h00. Inputs h00,h10,h01 only; fit mean interaction on development groups only.',
        hidden_layers='Index0 embedding; intermediate raw layer outputs; final index after final norm.',
        caveat='First-token yes/no checks measure only first-token format and answer, not complete generation or stopping. Development correction scores are in-sample.'))
    if means:np.savez(dest/'development_interaction.npz',**means)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','collect','analyze']);p.add_argument('--side',choices=['4B','14B'],default='4B');p.add_argument('--smoke',action='store_true');p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.action=='prepare':
        m=prepare();print(json.dumps(dict(n=len(m['rows']),self_test=self_test())))
    elif args.action=='collect':collect(args.side,args.smoke,args.limit)
    else:analyze(args.side,args.smoke)
