"""Compare same-budget B30 native, gradient and transcoder interventions."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import ttest_1samp
from transformers.cache_utils import DynamicCache

import attribute_binding_benchmark as B
import mlp_factor_steer_common as prior
from mlp_fidelity_v2_collect_v4 import greedy
from mlp_fidelity_v2_tail import capability_interval
from mlp_functional_trace import load_model
from mlp_read_route_causal import branch, changed
from shared.dist_protocol.mechanism_trace import GpuLease
from shared.dist_protocol.transcoder import TopKTranscoder

METHODS = ('native','attribution','transcoder')


def logits_gap(logits, rows, surfaces):
    return torch.stack([z[surfaces[r['target']]].max()-z[surfaces[r['contrast']]].max()
                        for z,r in zip(logits,rows)])


def batch_inputs(rows):
    x=torch.tensor([r['tokens'] for r in rows],device='cuda')
    j=torch.tensor([r['sites'] for r in rows],device='cuda')
    i=torch.arange(len(rows),device='cuda')[:,None]
    return x[:,0].contiguous(), x[:,1].contiguous(), (i,j)


@contextmanager
def collect(model, ij, with_gradient=False, full=False):
    layer=model.model.layers[B.BLOCK]
    bank={}

    def mlp_input(_, args):
        bank['x']=args[0][ij].detach().float()

    def down_input(_, args):
        bank['a']=args[0][ij].detach().float()

    def pre_norm(_, args):
        bank['residual']=args[0][ij].detach()

    def mlp_output(_, args, y):
        bank['y']=y[ij].detach()

    def block_output(_, args, y):
        if full:
            bank['hidden']=y.detach()
        if with_gradient:
            bank['grad_root']=y.detach().requires_grad_(True)
            return bank['grad_root']

    hooks=[layer.mlp.register_forward_pre_hook(mlp_input),layer.mlp.down_proj.register_forward_pre_hook(down_input),
           layer.post_attention_layernorm.register_forward_pre_hook(pre_norm),layer.mlp.register_forward_hook(mlp_output),
           layer.register_forward_hook(block_output)]
    try:
        yield bank
    finally:
        for hook in hooks:
            hook.remove()


def delta_bank(clean, donor, tc):
    return donor['a']-clean['a'], tc.encode(donor['x'])-tc.encode(clean['x'])


def edited_vectors(da, dz, native_weight, tc_weight, selection):
    return {method: torch.einsum('bsk,skd->bsd', source.gather(-1,ids.expand(len(source),-1,-1)),weight[ids])
            for method,source,weight in zip(METHODS,(da,da,dz),(native_weight,native_weight,tc_weight))
            for ids in (selection[method],)}


def fit(model, tc, data, out):
    device='cuda'
    weight=model.model.layers[B.BLOCK].mlp.down_proj.weight.detach().T.float()
    tc_weight=tc.W_dec*tc.y_scale
    rows=[r for r in data['rows'] if r['split']=='discovery']
    scores={m:[] for m in METHODS}; offsets=[]; effects=[]; pairs=[]
    for lo in range(0,len(rows),4):
        group=rows[lo:lo+4];x,other,ij=batch_inputs(group)
        with collect(model,ij,with_gradient=True) as clean:
            logits=model(x,use_cache=False,logits_to_keep=1).logits[:,0].float()
            margins=logits_gap(logits,group,data['surfaces'])
            gradient=torch.autograd.grad(margins.sum(),clean['grad_root'])[0][ij].detach().float()
        with torch.no_grad(), collect(model,ij) as donor:
            alt=model(other,use_cache=False,logits_to_keep=1).logits[:,0].float()
            da,dz=delta_bank(clean,donor,tc)
            scores['native'].append((da.abs()*weight.norm(dim=-1)).cpu())
            scores['attribution'].append((-da*torch.einsum('bsd,ud->bsu',gradient,weight)).cpu())
            scores['transcoder'].append((dz.abs()*tc_weight.norm(dim=-1)).cpu())
            effects.append((margins.detach()-logits_gap(alt,group,data['surfaces'])).cpu())
            offsets.extend([[B.LENGTH-1-s for s in r['sites']] for r in group])
            pairs.append((da.cpu(),dz.cpu()))
        if (lo+4)%32==0:
            print('discovery',lo+4,flush=True)
    ranking={m:torch.cat(v).mean(0).topk(B.ATOMS,dim=-1).indices for m,v in scores.items()}
    selected={m:v.to(device) for m,v in ranking.items()}
    stress={m:[] for m in METHODS}
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i,(da,dz) in enumerate(pairs):
            group=rows[4*i:4*i+4]
            vectors=edited_vectors(da.to(device),dz.to(device),weight,tc_weight,selected)
            for method,value in vectors.items():
                sign=torch.tensor([1. if r['target']<r['contrast'] else -1. for r in group],device=device)
                stress[method].append((value*sign[:,None,None]).cpu())
    frozen={'run_id':B.RUN,'prereg_sha256':prior.file_sha(out/'prereg.json'),'runner_sha256':prior.file_sha(__file__),
            'indices':{m:v.tolist() for m,v in ranking.items()}, 'n':len(rows),
            'mean_clean_minus_donor_margin':float(torch.cat(effects).mean()),
            'capability_offsets':np.median(offsets,axis=0).astype(int).tolist()}
    np.savez_compressed(out/'capability-vectors.npz',**{m:torch.cat(v).mean(0).numpy() for m,v in stress.items()})
    frozen['capability_vectors_sha256']=prior.file_sha(out/'capability-vectors.npz')
    prior.save(out/'frozen-candidates.json',frozen)
    print('Frozen candidate indices',prior.file_sha(out/'frozen-candidates.json'),flush=True)


def tail_cache(model, clean, hidden):
    cache=DynamicCache(config=model.config)
    for old,new in zip(clean.layers[:B.BLOCK+1],cache.layers):
        new.keys,new.values,new.is_initialized=old.keys,old.values,True
    pos=torch.arange(B.CUT,device=hidden.device)[None]
    rope=model.model.rotary_emb(hidden,pos)
    for block in model.model.layers[B.BLOCK+1:]:
        hidden=block(hidden,position_ids=pos,position_embeddings=rope,past_key_values=cache,use_cache=True)
    return cache


def prefix(model, x, ij, full):
    with collect(model,ij,full=full) as bank:
        cache=model.model(x[:,:B.CUT],use_cache=True).past_key_values
    return bank,cache


def suffix_logits(model, x, cache):
    try:
        return model(x[:,B.CUT:],past_key_values=cache,use_cache=True,logits_to_keep=1).logits[:,0].float()
    finally:
        cache.crop(B.CUT)


def changed_cache(model, clean, bank, ij, delta):
    changed_out=(bank['y'].float()+delta).to(bank['y'].dtype)
    hidden=bank['hidden'].clone()
    hidden[ij]=bank['residual']+changed_out
    actual=(changed_out.float()-bank['y'].float()).norm(dim=-1)
    return tail_cache(model,clean,hidden),actual


def control_transform(device, width):
    rng=np.random.default_rng(B.SEED+7)
    order=torch.tensor(rng.permutation(width),device=device)
    sign=torch.tensor(rng.choice([-1.,1.],width),device=device,dtype=torch.float32)
    return lambda value: value[...,order]*sign


def measures(logits, original, donor, rows, surfaces, tok, generated=None):
    lp=logits.log_softmax(-1)
    clean=original.log_softmax(-1);alt=donor.log_softmax(-1)
    kl=lambda p: (p.exp()*(p-lp)).sum(-1).clamp_min(0).tolist()
    top=logits.argmax(-1).tolist()
    if generated is None:
        generated=[[i] for i in top]
    return [{'gap':gap,'KL_clean':a,'KL_donor':b,'first_id':i,'tokens':seq,'text':tok.decode(seq),
             'word':B.first_word(tok.decode(seq)), 'first_target':i in surfaces[r['target']],
             'first_alternate':i in surfaces[r['contrast']]}
            for gap,a,b,i,seq,r in zip(logits_gap(logits,rows,surfaces).tolist(),kl(clean),kl(alt),top,generated,rows)]


def summarize(rows, gate):
    clean=np.array([r['clean']['word']==r['target'] for r in rows])
    result={'n':len(rows),'clean_word':int(clean.sum()),'methods':{}}
    clean_alt=np.array([r['clean']['word']==r['contrast'] for r in rows])
    for method in METHODS:
        tests=[r['methods'][method] for r in rows]
        differences=np.array([r['matched']['gap']-r['doses']['1.0']['gap'] for r in tests])
        test=ttest_1samp(differences,0.,alternative='greater')
        p=float(test.pvalue) if np.isfinite(test.pvalue) else 1.
        ci=test.confidence_interval(.95)
        word=np.array([r['doses']['1.0']['word']==g['contrast'] for r,g in zip(tests,rows)])
        ctl=np.array([r['matched']['word']==g['contrast'] for r,g in zip(tests,rows)])
        replay=sum(t['replay_exact'] for t in tests);rescue=sum(t['rescue_exact'] for t in tests)
        pre=max(t['pre_error'] for t in tests);post=max(t['post_error'] for t in tests)
        primary=bool(clean.mean()>=gate['base_clean_first_word'] and p<gate['alpha_one_sided'] and
                     differences.mean()>=gate['min_margin_effect'] and pre<=1e-5 and post<=.05)
        result['methods'][method]={'margin_over_control':float(differences.mean()),'p_one_sided':p,
            'margin_lower_95':float(ci.low) if np.isfinite(ci.low) else None,
            'alternate_word':int(word.sum()),'matched_alternate_word':int(ctl.sum()),'clean_alternate_word':int(clean_alt.sum()),
            'replay_exact':replay,'restore_exact':rescue,'pre_norm_error':pre,'post_norm_error':post,
            'primary_gate':primary,'anchor':bool(primary and (word.mean()-clean_alt.mean())>=.10 and
                        (word.mean()-ctl.mean())>=.05 and replay==rescue==len(rows)),
            'dose_curve':{str(dose):{key:float(np.mean([r['doses'][str(dose)][key] for r in tests]))
                            for key in ('gap','KL_clean','KL_donor')} for dose in B.DOSES}}
    return result


@torch.inference_mode()
def evaluate(model,tok,eos,tc,data,out,split):
    frozen=prior.read(out/'frozen-candidates.json');gate=prior.read(out/'prereg.json')['gate']
    assert frozen['runner_sha256']==prior.file_sha(__file__)
    selection={m:torch.tensor(i,device='cuda') for m,i in frozen['indices'].items()}
    weight=model.model.layers[B.BLOCK].mlp.down_proj.weight.detach().T.float()
    control=control_transform('cuda',weight.shape[-1]);tc_weight=tc.W_dec*tc.y_scale
    rows=[r for r in data['rows'] if r['split']==split];record=[];canary=None
    started=time.monotonic()
    for lo in range(0,len(rows),4):
        batch=rows[lo:lo+4];x,other,ij=batch_inputs(batch)
        clean,cache=prefix(model,x,ij,True)
        donor,donor_cache=prefix(model,other,ij,False)
        z,g=branch(model,x,cache,eos);dz=suffix_logits(model,other,donor_cache)
        del donor_cache
        if canary is None:
            bypass=tail_cache(model,cache,clean['hidden']);test,tokens=branch(model,x,bypass,eos)
            canary={'max_logit_difference':float((test-z).abs().max()),'exact':bool(torch.equal(test,z) and tokens==g)}
            assert canary['exact'],canary
            del bypass
        vectors=edited_vectors(*delta_bank(clean,donor,tc),weight,tc_weight,selection)
        observations=[{'id':r['id'],'target':r['target'],'contrast':r['contrast'],'clean':c,'methods':{}}
            for r,c in zip(batch,measures(z,z,dz,batch,data['surfaces'],tok,g))]
        for method,vec in vectors.items():
            dmetrics={};primary=None
            for dose in B.DOSES:
                edited,realized=changed_cache(model,cache,clean,ij,vec*dose)
                if dose==1.:
                    altered,tokens=branch(model,x,edited,eos)
                    forward,reverse,_=changed(cache,edited)
                    replay,rseq=branch(model,x,cache,eos,forward);restore,sseq=branch(model,x,edited,eos,reverse)
                    exact_replay=((altered==replay).all(-1).tolist(),[a==b for a,b in zip(tokens,rseq)])
                    exact_restore=((z==restore).all(-1).tolist(),[a==b for a,b in zip(g,sseq)])
                    primary=realized
                else:
                    altered=suffix_logits(model,x,edited);tokens=None
                dmetrics[str(dose)]=measures(altered,z,dz,batch,data['surfaces'],tok,tokens)
                del edited
            matched=control(vec)
            denominator=vec.norm(dim=-1).clamp_min(1e-20)
            pre=((matched.norm(dim=-1)-vec.norm(dim=-1)).abs()/denominator).amax(-1).tolist()
            matched_cache,applied=changed_cache(model,cache,clean,ij,matched)
            control_logits,control_gen=branch(model,x,matched_cache,eos)
            post=torch.where(primary>1e-3,(applied-primary).abs()/primary.clamp_min(1e-20),0).amax(-1).tolist()
            controls=measures(control_logits,z,dz,batch,data['surfaces'],tok,control_gen)
            for j,row in enumerate(observations):
                row['methods'][method]={'doses':{k:v[j] for k,v in dmetrics.items()},'matched':controls[j],
                    'pre_error':pre[j],'post_error':post[j], 'applied_norm':primary[j].tolist(),
                    'replay_exact':exact_replay[0][j] and exact_replay[1][j],
                    'rescue_exact':exact_restore[0][j] and exact_restore[1][j]}
            del matched_cache
        record.extend(observations)
        if lo==0 or (lo+4)%32==0:
            print(split,lo+4,'/',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
    summary=summarize(record,gate)
    prior.write_gz(out/f'{split}.json.gz',{'rows':record,'summary':summary,'canary':canary,
                   'seconds':time.monotonic()-started,'frozen_candidates_sha256':prior.file_sha(out/'frozen-candidates.json')})
    print(split,summary,flush=True)


@torch.inference_mode()
def capability(model,tok,out):
    frozen=prior.read(out/'frozen-candidates.json');data=prior.read(prior.CAP)
    assert sum(r['gate_eligible'] for r in data)==2200
    vector_file=out/'capability-vectors.npz'
    assert prior.file_sha(vector_file)==frozen['capability_vectors_sha256']
    with np.load(vector_file) as arrays:
        directions={m:torch.from_numpy(arrays[m]).to('cuda') for m in METHODS}
    layer=model.model.layers[B.BLOCK].mlp
    labels=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCDE'];results=[]
    for lo in range(0,len(data),12):
        rows=data[lo:lo+12]
        prompts=[tok.encode(tok.apply_chat_template([{'role':'user','content':r['question']+'\n'+
                 '\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。'}],
                 tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for r in rows]
        inputs={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()}
        width=inputs['input_ids'].shape[1]
        loc=[(i,s,width-1-d) for i,q in enumerate(prompts) for s,d in enumerate(frozen['capability_offsets']) if d<len(q)]
        i,s,j=torch.tensor(loc,device='cuda',dtype=torch.long).reshape(-1,3).T
        sign=torch.tensor([1. if int(prior.digest([B.SEED,lo+v])[:8],16)%2 else -1. for v in range(len(rows))],device='cuda')
        logits={'clean':model(**inputs,use_cache=False,logits_to_keep=1).logits[:,0]}
        for method,vec in directions.items():
            def replace(_,args,y):
                yy=y.clone();yy[i,j]=(y[i,j].float()+vec[s]*sign[i,None]).to(y.dtype);return yy
            hook=layer.register_forward_hook(replace)
            try:
                logits[method]=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,0]
            finally:
                hook.remove()
        for k,r in enumerate(rows):
            results.append({'id':r.get('id'),'source':r.get('source'),'eligible':r['gate_eligible'],'answer':r['answer'],
                'choice':{m:'ABCDE'[int(z[k,labels[:len(r['choices'])]].argmax())] for m,z in logits.items()}})
        if (lo+12)%240==0:
            print('capability',lo+12,flush=True)
    eligible=np.asarray([r['eligible'] for r in results]);correct=lambda m:[r['choice'][m]==r['answer'] for r in results]
    summary={m:capability_interval(correct('clean'),correct(m),eligible) for m in METHODS}
    prior.write_gz(out/'capability.json.gz',{'scope':'Fixed discovery mean, median source offsets, unrelated archived MCQ pool.',
        'source_sha256':prior.file_sha(prior.CAP),'summary':summary,'rows':results})
    print(summary,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','evaluate','capability']);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();out=a.out
    torch.set_num_threads(4)
    prereg=prior.read(out/'prereg.json')
    assert prior.file_sha(B.__file__)==prereg['runner_sha256']
    assert prior.file_sha(out/'inputs.json.gz')==prereg['inputs_sha256']
    assert prior.file_sha(B.TC/'transcoder.npz')==prereg['transcoder_sha256']
    data=prior.read_gz(out/'inputs.json.gz')
    with GpuLease():
        model,tok,eos=load_model()
        if a.stage=='capability':
            capability(model,tok,out);return
        with np.load(B.TC/'transcoder.npz') as arrays:
            tc=TopKTranscoder.from_state(arrays,device='cuda')
        if a.stage=='fit':
            fit(model,tc,data,out)
        else:
            for split in list(B.COUNTS)[1:]:
                if not (out/f'{split}.json.gz').exists():
                    evaluate(model,tok,eos,tc,data,out,split)


if __name__=='__main__':
    main()
