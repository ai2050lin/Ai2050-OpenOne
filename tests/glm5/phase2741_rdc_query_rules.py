"""Available-history predictors: no true queried future state enters a feature.

The native block is a mechanistic feature map, not an entire-model replay.
Query-only prototypes are frozen before fitting and are a disclosed input cost.
"""
import argparse
from collections import defaultdict
from rdc_query_common import *

NAMES=['query_only','uniform','quadratic','shuffled_values','ordered_softmax']
OUT=BASE/'rules'

def freeze():
    p=OUT/'protocol.json'
    if p.exists():return read(p)
    material=read(BASE/'material/protocol.json')
    value={'timestamp':stamp(),'source':snapshot(__file__),'sample_ids':material['detailed_prefix_ids'],
      'candidates':NAMES,'native_block':12,'targets':['rawH24','rawH36','postnorm'],
      'available_inputs':['Observed prefix H12 and block12 KV','Known query string alone H12/Q/K/V/H13','Original block12 parameters'],
      'forbidden_inputs':['Actual queried H12/H24/H36/postnorm','Future token','Answer label','Heldout fitted statistics'],
      'attention':'Query-alone last Q and all query-alone KV shifted to true absolute positions; exact native prefix KV. This is an approximation to context-conditioned query construction.',
      'numerics':'Scores and softmax FP32, weights cast to original BF16 before V and Wo; original BF16 MLP12. No state-coordinate reduction.',
      'controls':'Uniform and Taylor2 keep the same values/parameters; shuffled past V cyclically permutes source positions without changing K; query-only ignores prefix in native feature.',
      'fit':'Same per-coordinate four-column decoder [1,prefixH12last,queryOnlyH13last,candidateH13last]; standardize using training only; L2 coefficients with unpenalized intercept.',
      'lambdas':[0.0001,0.01,1.,100.],
      'split':'Fit train-source/train-query only; choose lambda on validation-source/validation-query; report source/query four-way generalization and source-cluster paired intervals.',
      'capacity_caveat':'Equal nominal coefficients, not equal effective rank: query-only duplicates one regressor. Native nonlinear feature uses known original parameters.',
      'resource':'576*100*5 full2560-coordinate candidates retained as original BF16 bits; noPCA/TopK. Fixed query full-model prototype cost recorded separately.',
      'selection_rule':'Retain ordered prediction as useful only when heldout source+query postnorm error improves over uniform and shuffled controls with positive source-cluster paired95% intervals. This is not semantic closure.'}
    immutable(p,value);return value

def tensor(a,device='cuda'):
    import torch
    return torch.from_numpy(unbits(a).copy()).to(device=device,dtype=torch.bfloat16)

def rotate(x,cos,sin):
    import torch
    h=x.shape[-1]//2
    return x*cos[None]+torch.cat([-x[...,h:],x[...,:h]],-1)*sin[None]

def prototype_native(model,pr,probes):
    """Recreate unrotated native K with the original exact-length batch shapes."""
    import torch
    layer=model.model.layers[12];att=layer.self_attn;groups=defaultdict(list);result={};checks=[]
    for i,p in enumerate(probes):groups[len(p['token_ids'])].append(i)
    for length,indices in sorted(groups.items()):
      for j0 in range(0,len(indices),16):
        batch=indices[j0:j0+16];h=tensor(np.stack([pr[f'p{q}_H12'] for q in batch]));z=layer.input_layernorm(h)
        shape=(len(batch),length,-1,128)
        qq=att.q_norm(att.q_proj(z).view(shape)).transpose(1,2)
        kk=att.k_norm(att.k_proj(z).view(shape)).transpose(1,2)
        vv=att.v_proj(z).view(shape).transpose(1,2)
        pos=torch.arange(length,device='cuda')[None].expand(len(batch),-1)
        cos,sin=model.model.rotary_emb(h,pos)
        rr=kk*cos[:,None]+torch.cat([-kk[...,64:],kk[...,:64]],-1)*sin[:,None]
        for j,q in enumerate(batch):
            savedq=tensor(pr[f'p{q}_q_before_rope']).transpose(0,1)
            ck={'q':q,'Q_bit_equal':bool(torch.equal(qq[j],savedq)),
              'K_rope_bit_equal':bool(torch.equal(rr[j],tensor(pr[f'p{q}_keys_after_rope']))),
              'V_bit_equal':bool(torch.equal(vv[j],tensor(pr[f'p{q}_values'])))}
            assert all(ck[k] for k in ['Q_bit_equal','K_rope_bit_equal','V_bit_equal']),ck
            checks.append(ck);result[q]=(qq[j].clone(),kk[j].clone(),vv[j].clone(),h[j,-1].clone())
    return result,checks

def merge_check(q,k,v):
    """Independent FP64 stable numerator/denominator recurrence at fixed q."""
    import torch
    s=(q.double().unsqueeze(-2)@k.double().transpose(-2,-1)).squeeze(-2)/np.sqrt(128)
    direct=(s.softmax(-1).unsqueeze(-2)@v.double()).squeeze(-2)
    m=torch.full((s.shape[0],),-torch.inf,device=s.device,dtype=torch.float64)
    z=torch.zeros_like(m);num=torch.zeros_like(direct)
    for ix in torch.tensor_split(torch.arange(s.shape[-1],device=s.device),4):
        if not len(ix):continue
        ss=s[:,ix];local=ss.max(-1).values;new=torch.maximum(m,local)
        zz=(ss-new[:,None]).exp();scale=(m-new).exp()
        num=num*scale[:,None]+(zz.unsqueeze(-2)@v[:,ix].double()).squeeze(-2)
        z=z*scale+zz.sum(-1);m=new
    return float((direct-num/z[:,None]).abs().max())

def rule_features(model,protos,pr,kp,vp,verifymerge=False):
    import torch
    layer=model.model.layers[12];att=layer.self_attn;n=kp.shape[-2];pred=np.empty((5,100,2560),np.uint16);merges=[]
    for q in range(100):
        qq,kk,vv,h=protos[q];length=qq.shape[-2]
        pos=torch.arange(n,n+length,device='cuda')[None];cos,sin=model.model.rotary_emb(h[None,None],pos)
        qr=rotate(qq,cos[0],sin[0])[:,-1];kr=rotate(kk,cos[0],sin[0]).repeat_interleave(4,dim=0);vr=vv.repeat_interleave(4,dim=0)
        k=torch.cat([kp,kr],-2);v=torch.cat([vp,vr],-2)
        scores=(qr.float().unsqueeze(-2)@k.float().transpose(-2,-1)).squeeze(-2)*att.scaling
        soft=scores.softmax(-1);poly=1+scores+0.5*scores.square();poly=poly/poly.sum(-1,keepdim=True)
        weights=[torch.ones_like(soft)/soft.shape[-1],poly,soft,soft];shuffled=torch.cat([vp.roll(1,dims=-2),vr],-2);readouts=[]
        for j,w in enumerate(weights):
            val=shuffled if j==2 else v;rv=(w.to(torch.bfloat16).unsqueeze(-2)@val).squeeze(-2).reshape(1,1,-1);readouts.append(att.o_proj(rv)[0,0])
        stacked=torch.stack(readouts)+h[None];y=stacked+layer.mlp(layer.post_attention_layernorm(stacked))
        pred[0,q]=pr[f'p{q}_H13'][-1];pred[1:,q]=bits(y)
        if verifymerge and q in [0,50,99]:merges.append({'probe_index':q,'max_abs_error':merge_check(qr,k,v)})
    return pred,merges

def capture():
    import torch
    protocol=freeze();ids=set(protocol['sample_ids']);rows=[r for r in gzread(BASE/'material/natural.json.gz') if r['sample_id'] in ids]
    assert len(rows)==576
    for r in rows:assert (BASE/'capture/commits'/f"{r['sample_id']}.json").exists()
    if (OUT/'capture_result.json').exists():print('QUERY_RULES_ALREADY_COMPLETE');return
    start=time.monotonic();guard(1600*1024**2);model=None
    try:
      model,tok=load('qwen4',OUT)
      probes=read(BASE/'probes/protocol.json')['probes']
      with np.load(BASE/'prototypes/qwen4.npz') as z:pr={k:z[k].copy() for k in z.files if k!='logprobs'}
      merges=[];commits=[]
      with torch.inference_mode():
        protos,checks=prototype_native(model,pr,probes)
        for i,row in enumerate(rows):
            sid=row['sample_id'];file=OUT/'features'/f'{sid}.npz';commit=OUT/'commits'/f'{sid}.json'
            if commit.exists():
                c=read(commit);assert sha(file)==c['sha256'];commits.append(c);continue
            with np.load(BASE/'capture/fields'/f'{sid}.npz') as z:
                kp=tensor(z['prefix_block12_keys']).repeat_interleave(4,dim=0)
                vp=tensor(z['prefix_block12_values']).repeat_interleave(4,dim=0)
            pred,checked=rule_features(model,protos,pr,kp,vp,verifymerge=i<9);merges.extend(dict(r,sample_id=sid) for r in checked)
            assert np.isfinite(unbits(pred)).all();npz(file,candidate_H13=pred)
            c={'timestamp':stamp(),'sample_id':sid,'sha256':sha(file),'shape':list(pred.shape),
              'input_capture_sha256':read(BASE/'capture/commits'/f'{sid}.json')['array_sha256'],'no_future_target_read':True}
            save(commit,c);commits.append(c)
            if (i+1)%12==0:print('QUERY_RULE_FEATURES',i+1,len(rows),'seconds',round(time.monotonic()-start,1),flush=True);guard()
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'rows':len(commits),'queries':57600,
          'prototype_reconstruction':checks,'fixed_query_stable_merge':merges,
          'merge_max_abs_error':max(r['max_abs_error'] for r in merges),'seconds':time.monotonic()-start,
          'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
          'meaning':'Fixed-query stable composition is an attention identity. Native ordered candidate differs from actual query response because early query Q/K/V are context-free prototypes.'}
        assert result['merge_max_abs_error']<1e-10
        save(OUT/'capture_result.json',result);ledger('query_rule_features',result['seconds']);print('QUERY_RULE_FEATURES_DONE',result['seconds'],flush=True)
    except Exception as exc:failure(OUT,start,exc);raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print('RULE_PROTOCOL_FROZEN',len(freeze()['sample_ids']))
    else:capture()
