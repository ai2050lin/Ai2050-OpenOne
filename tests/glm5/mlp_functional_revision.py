"""Revalidate frozen writers on new materials, then run real scope interventions."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import time

import numpy as np
import torch
from scipy.stats import t as student_t
from torch.nn.functional import linear

import mlp_functional_common as C
from mlp_fidelity_v2_collect_v4 import first_word, greedy
from mlp_fidelity_v2_tail import capability_interval
from mlp_functional_causal import edited, summarize_controlled
from mlp_functional_trace import bit_array, cells, contrasts, load_model
from shared.dist_protocol.mechanism_trace import GpuLease
from shared.dist_protocol.natural_register_runner import snapshot
from shared.dist_protocol.transcoder import TopKTranscoder

OUT=C.OUT/'revision'
ATTRS=['location','industry']


def frozen():
    corpus=C.read(OUT/'inputs.json');prereg=C.read(OUT/'prereg.json');path=C.OUT/'path-analysis.json'
    assert C.digest(corpus)==prereg['corpus_sha256'] and C.file_sha(path)==prereg['frozen_v1_paths_sha256']
    return corpus['rows'],prereg,C.read(path)


def decode(tok,tokens):
    text=tok.decode(tokens,skip_special_tokens=True)
    return text,first_word(text)


def observe(part):
    result=OUT/f'{part}-result.json'
    if result.exists():return
    if part=='confirm':assert C.read(OUT/'pilot-result.json')['passed']
    rows,plan,paths=frozen();torch.set_num_threads(4)
    contract={**C.identity(),'schema':'functional_revision_observation.v2','part':part,'prereg_sha256':C.file_sha(OUT/'prereg.json'),
              'source_sha256':C.file_sha(__file__),'pilot_gate_passed':part=='confirm'}
    C.save(OUT/f'{part}-execution-prereg.json',contract)
    lo,hi=(0,32) if part=='pilot' else (32,352)
    outcomes=[];samples=[];points=[];raw=[];started=time.time()
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();ctx={};handles=[]
        directions=torch.tensor([paths['directions'][a]['direction'] for a in ATTRS],device='cuda')
        candidate_blocks=sorted({p['block'] for k in ATTRS for p in paths['selected'][k]})
        native_heads={};native_mlp={}
        if part=='confirm':
            def r_guard(_,args):
                if ctx['capture']:ctx['incoming']=args[0][:,ctx['position']].detach().clone()
            handles.append(model.model.layers[35].post_attention_layernorm.register_forward_pre_hook(r_guard))
            for b in candidate_blocks:
                layer=model.model.layers[b]
                def h_guard(_,args,b=b):
                    if ctx['capture']:native_heads[b]=args[0][:,ctx['position']].reshape(4,32,128).detach().clone()
                def m_guard(_,args,value,b=b):
                    if ctx['capture']:native_mlp[b]=value[:,ctx['position']].detach().clone()
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(h_guard))
                if b<35:handles.append(layer.mlp.register_forward_hook(m_guard))
        coefficients={b:(directions@model.model.layers[b].self_attn.o_proj.weight.float()).reshape(2,32,128) for b in candidate_blocks}
        for b in range(lo,hi):
            old=None
            for q in [0,1]:
                batch=rows[b*8+q:(b+1)*8:2];pos=batch[0]['prequestion']
                ids=torch.tensor([r['tokens'] for r in batch],device='cuda');ctx.update(position=pos,capture=part=='confirm')
                assert all(r['prequestion']==pos for r in batch)
                o=model(input_ids=ids,use_cache=True,logits_to_keep=1);ctx['capture']=False
                if part=='confirm':
                    if q==0:
                        old=ctx['incoming'];scores=[]
                        for a,attr in enumerate(ATTRS):
                            cols=[]
                            for p in paths['selected'][attr]:
                                layer=p['block'];head=p['head']
                                values=(native_mlp[layer].float()@directions[a] if head is None else
                                        native_heads[layer][:,head].float()@coefficients[layer][a,head])
                                cols.append(values)
                            scores.append(torch.stack(cols,dim=-1))
                        points.append(torch.stack(scores,1).cpu().numpy())
                        samples.append((old.float()@directions.T).cpu().numpy())
                        raw.append({'incoming':bit_array(old),'embedding':bit_array(model.model.embed_tokens(ids[:,pos])),
                                    **{f'head{layer}':bit_array(v) for layer,v in native_heads.items()},
                                    **{f'mlp{layer}':bit_array(v) for layer,v in native_mlp.items()}})
                    else:assert torch.equal(old,ctx['incoming']),('future question changed incoming B35',b)
                for row,toks in zip(batch,greedy(model,o.past_key_values,o.logits[:,-1],eos,limit=4)):
                    text,word=decode(tok,toks)
                    outcomes.append({k:v for k,v in row.items() if k!='tokens'}|{'generated':text,'word':word,'generated_ids':toks,'correct':word==row['expected']})
                del o
            if (b+1)%16==0:print(part,b+1,'/',hi,'seconds',round(time.time()-started,1),flush=True)
        for h in handles:h.remove()
        assert snapshot(C.MODEL)==contract['model_snapshot']
    by_cell=cells(outcomes);ncell,threshold=(16,14) if part=='pilot' else (80,72)
    assert len(by_cell)==(16 if part=='pilot' else 32) and all(v['n']==ncell for v in by_cell)
    passed=all(v['correct']>=threshold for v in by_cell)
    replication={};meta=None
    if part=='confirm':
        ss=np.asarray(samples);pp=np.asarray(points);fields={k:np.concatenate([r[k] for r in raw]) for k in raw[0]}
        f=OUT/'observed-native.npz'
        np.savez(f,**fields,writer_scores=pp,incoming_scores=ss)
        meta={'path':'revision/observed-native.npz','sha256':C.file_sha(f),'fields':{k:list(v.shape) for k,v in fields.items()},'boundaries':'actual same prequestion token; Q0 only, incoming B35 bitwise checked across both questions'}
        for a,attr in enumerate(ATTRS):
            labels=np.asarray([0,1,0,1] if a==0 else [0,1,1,0],bool)
            cutoff=paths['directions'][attr]['threshold'];report={}
            picks=[{**p,'revision':{}} for p in paths['selected'][attr]]
            for split,sl in [('confirm',slice(0,160)),('lockbox',slice(160,320))]:
                r=ss[sl,:,a];d=contrasts(pp[sl,:,a],attr);report[split]={'classification_accuracy':float(np.mean((r>cutoff)==labels)),
                    'paired_accuracy_by_assignment':(contrasts(r,attr)>0).mean(0).tolist()}
                for j,p in enumerate(picks):
                    values=d[:,:,j];mean=values.mean(0);hw=student_t.ppf(1-.05/64,159)*values.std(0,ddof=1)/np.sqrt(160)
                    p['revision'][split]={'mean_by_R':mean.tolist(),'CI_by_R':np.stack([mean-hw,mean+hw],-1).tolist(),'passed':bool(np.all(mean-hw>0))}
            for p in picks:p['replicated']=passed and all(r['passed'] for r in p['revision'].values())
            replication[attr]={'incoming':report,'writers':picks}
    C.save(result,{**contract,'status':'complete','passed':passed,'cells':by_cell,'rows':outcomes,'replication':replication,
                   'native':meta,'seconds':time.time()-started})
    print(part,'behavior gate',passed,'min correct',min(v['correct'] for v in by_cell),flush=True)
    if replication:print({k:[(p['block'],p['head'],p['replicated']) for p in v['writers']] for k,v in replication.items()},flush=True)


@contextmanager
def matched(model,indices,pos,heads,mlps):
    """Subtract a fixed orthogonal scrambling of each writer, at the same layer and position."""
    handles=[]
    try:
        for b,h in heads:
            def replace(m,args,o,b=b,h=h):
                sl=slice(h*128,(h+1)*128);x=args[0][indices,pos,sl]
                writer=linear(x.float(),m.weight[:,sl].float())
                out=o.clone();out[indices,pos]=(o[indices,pos].float()-scramble(writer,b,h)).to(o.dtype)
                return out
            handles.append(model.model.layers[b].self_attn.o_proj.register_forward_hook(replace))
        for b in mlps:
            def replace(_,args,o,b=b):
                writer=o[indices,pos].float();out=o.clone()
                out[indices,pos]=(writer-scramble(writer,b,32)).to(o.dtype)
                return out
            handles.append(model.model.layers[b].mlp.register_forward_hook(replace))
        yield
    finally:
        for h in handles:h.remove()


def scramble(x,b,h):
    rng=np.random.default_rng(C.SEED+b*71+h)
    ix=torch.from_numpy(rng.permutation(x.shape[-1])).to(x.device)
    signs=torch.from_numpy(rng.choice(np.array([-1,1],np.float32),size=x.shape[-1])).to(x.device)
    return x[:,ix]*signs


def causal(part):
    result=OUT/f'{part}-causal.json'
    if result.exists():return
    rows,plan,paths=frozen();obs=C.read(OUT/'confirm-result.json');torch.set_num_threads(4)
    picks=paths['selected']['industry'];heads=[(r['block'],r['head']) for r in picks if r['head'] is not None]
    mlps=[r['block'] for r in picks if r['head'] is None];h2=[(34,2)];assert h2[0] in heads
    offsets=Counter(r['assistant']-r['prequestion'] for r in rows if r['split']=='pilot');offset=offsets.most_common(1)[0][0]
    if part=='generated':
        selected={b:int(C.digest([C.SEED,b,'revision-causal'])[:8],16)%4 for b in range(192,352)}
        items=[r for r in rows if r['b'] in selected and r['r']*2+r['t']==selected[r['b']]];assert len(items)==320
        ops=[('clean',[],[],0.),('joint_0.5',heads,mlps,.5),('joint_1',heads,mlps,1.),('attention_1',heads,[],1.),
             ('mlp_1',[],mlps,1.),('head34_2',h2,[],1.),('random_1',[],[],0.)]
    else:
        items=C.read(C.CAP);assert sum(r['gate_eligible'] for r in items)==2200
        ops=[('clean',[],[],0.),('functional_sae',[],[],0.)]
        if obs['passed']:ops.append(('upstream_joint',heads,mlps,1.))
    trained=C.read(C.OUT/'dictionary-training.json')['outputs']['functional_sae']
    contract={**C.identity(),'schema':'functional_revision_intervention.v2','part':part,'source_sha256':C.file_sha(__file__),
              'prereg_sha256':C.file_sha(OUT/'prereg.json'),'observation_sha256':C.file_sha(OUT/'confirm-result.json'),
              'operations':ops,'n':len(items),'behavior_gate':obs['passed'],'capability_offset_from_assistant':offset,
              'dictionary_sha256':trained['sha256'],'capability_input_sha256':C.file_sha(C.CAP),'matched_control':'same-layer writer signed permutation, equal source-vector Euclidean norm before BF16 add'}
    C.save(OUT/f'{part}-execution-prereg.json',contract)
    if part=='generated' and not obs['passed']:
        C.save(result,{**contract,'status':'not_eligible','summary':{},'rows':[]});return
    output=[];canary=[];started=time.time();progress=OUT/f'{part}-progress.jsonl'
    if progress.exists():output=[json.loads(x) for x in progress.read_text(encoding='utf-8').splitlines()]
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();bs=4 if part=='generated' else 12
        d=torch.tensor([paths['directions'][a]['direction'] for a in ATTRS],device='cuda')
        cutoff=torch.tensor([paths['directions'][a]['threshold'] for a in ATTRS],device='cuda')
        word_ids={w:sorted({tok.encode(f,add_special_tokens=False)[0] for f in [w,w.capitalize(),' '+w,' '+w.capitalize()]}) for w in ['coastal','inland','fishing','mining']}
        letter_ids=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABCDE']
        dictionary=None
        if part=='capability':
            p=C.OUT/trained['path'];assert C.file_sha(p)==trained['sha256']
            with np.load(p) as weights:dictionary=TopKTranscoder.from_state(weights,device='cuda')
        for lo in range(len(output),len(items),bs):
            batch=items[lo:lo+bs];indices=torch.arange(len(batch),device='cuda')
            if part=='generated':
                inputs={'input_ids':torch.tensor([r['tokens'] for r in batch],device='cuda')}
                pos=torch.tensor([r['prequestion'] for r in batch],device='cuda')
            else:
                prompts=[]
                for r in batch:
                    text=r['question']+'\n'+'\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。'
                    prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                           add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
                inputs={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()}
                full=inputs['input_ids'].shape[1]
                pos=torch.tensor([max(full-len(v),full-1-offset) for v in prompts],device='cuda')
            per={};clean_lp=None
            for name,h,m,alpha in ops:
                replace=dictionary if name=='functional_sae' else None
                p=torch.full_like(pos,inputs['input_ids'].shape[1]-1) if replace is not None else pos
                with edited(model,indices,p,h,m,alpha,part=='generated',replace) as collected:
                    if name=='random_1':
                        with matched(model,indices,p,heads,mlps):o=model(**inputs,use_cache=True,logits_to_keep=1)
                    else:o=model(**inputs,use_cache=part=='generated',logits_to_keep=1)
                logits=o.logits[:,-1].float()
                if lo==0 and name=='clean':
                    with edited(model,indices,pos,heads,mlps,0.,False):ref=model(**inputs,use_cache=part=='generated',logits_to_keep=1).logits[:,-1].float()
                    assert torch.equal(logits,ref);canary.append({'zero_dose_logit_bitwise_equal':True})
                per[name]=[]
                if part=='generated':
                    lp=logits.log_softmax(-1)
                    if clean_lp is None:clean_lp=lp
                    kl=(clean_lp.exp()*(clean_lp-lp)).sum(-1).cpu().tolist();margins=(collected[0]@d.T-cutoff).cpu().numpy()
                    for j,(row,ids) in enumerate(zip(batch,greedy(model,o.past_key_values,logits,eos,limit=4))):
                        text,word=decode(tok,ids)
                        gap=float(lp[j,word_ids[row['expected']]].max()-lp[j,word_ids[row['contrast']]].max())
                        per[name].append({'generated':text,'word':word,'generated_ids':ids,'correct':word==row['expected'],'KL':kl[j],'answer_gap':gap,
                                          'signed_location_residual':float(margins[j,0]*(2*row['t']-1)),'signed_industry_residual':float(margins[j,1]*(2*row['i']-1))})
                else:
                    for j,row in enumerate(batch):
                        pred='ABCDE'[int(logits[j,letter_ids[:len(row['choices'])]].argmax())]
                        per[name].append({'prediction':pred,'correct':pred==row['answer'],'offset':int(inputs['input_ids'].shape[1]-1-p[j])})
                del o
            records=[]
            for j,row in enumerate(batch):
                item={k:v for k,v in row.items() if k!='tokens'} if part=='generated' else {k:row.get(k) for k in ['id','source','answer','gate_eligible']}
                records.append({**item,'index':lo+j,'conditions':{k:v[j] for k,v in per.items()}})
            output.extend(records)
            with progress.open('a',encoding='utf-8') as f:
                f.writelines(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n' for v in records)
            if (lo+len(batch))%(32 if part=='generated' else 120)==0:print(part,lo+len(batch),'/',len(items),'seconds',round(time.time()-started,1),flush=True)
        assert snapshot(C.MODEL)==contract['model_snapshot']
    if part=='generated':
        summary=summarize_controlled(output,False);s=summary['industry'];joint=s['operations']['joint_1']
        s['passed']=joint['loss']['mean']>=.10 and joint['mcnemar_exact_p']<.05 and s['target_minus_random']['mean']>=.05
        summary['primary_attribute']='industry'
    else:
        mask=np.asarray([r['gate_eligible'] for r in output],bool);base=[r['conditions']['clean']['correct'] for r in output]
        summary={name:capability_interval(base,[r['conditions'][name]['correct'] for r in output],mask) for name,*_ in ops if name!='clean'}
        summary['reference_only_n']=int((~mask).sum())
    C.save(result,{**contract,'status':'complete','rows':output,'summary':summary,'canaries':canary,'seconds':time.time()-started})
    print(part,summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['pilot','confirm','generated','capability']);p=parser.parse_args().part
    observe(p) if p in ['pilot','confirm'] else causal(p)
