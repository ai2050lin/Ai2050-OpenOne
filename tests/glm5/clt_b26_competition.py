"""Select native Q/K columns by predicted effects on competition among records."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import clt_b26_three_cues_v3 as previous
import clt_b26_three_cues as B
from deploy.storage_policy import check_write
from source_binding_design import read,sha

REFERENCE=B.OUT
B.OUT=B.HERE/'result/clt_b26_competition_v1'
B.TMP=B.ROOT/'tests/glm5_temp/clt_b26_competition_v1'
B.SEED=202611190123
B.METHODS=[*B.METHODS,('prior',512)]
B.configure()


def freeze():
    check_write(B.OUT,192<<20,operation='Competition campaign')
    old=read(REFERENCE/'delivery.json')
    assert sha(REFERENCE/'predictor.npz')==old['files']['predictor.npz']
    original=B.common.commit
    def commit(name,data):
        if name=='prereg.json':
            data['code'][Path(__file__).relative_to(B.ROOT).as_posix()]=sha(Path(__file__))
            path=Path(__file__).with_suffix('.md')
            data['code'][path.relative_to(B.ROOT).as_posix()]=sha(path)
            data['reference_predictor_sha']=sha(REFERENCE/'predictor.npz')
            data['protocol_change']='Same fresh first-slot families with record-contrast sensitivity rank in place of input variance rank. Equal budget prior ranking is a diagnostic arm.'
        original(name,data)
    B.common.commit=commit
    try:previous.freeze()
    finally:B.common.commit=original


def discovery():
    import torch as t
    from source_binding_runtime import Runtime
    pre=B.common.checked(); rows=B.materials('discovery',pre)
    assert sha(REFERENCE/'predictor.npz')==pre['reference_predictor_sha']
    with np.load(REFERENCE/'predictor.npz') as old:
        means={key:old['mean_'+key].copy() for key in ('q','k')}
    sums=np.zeros((2,5120),np.float64); counts=[]; start=0
    original=Runtime.block

    def block(rt,number):
        nonlocal start
        result=original(rt,number)
        if number!=B.BLOCK:return result
        att=result.self_attn;state={}; base={k:t.tensor(v,device='cuda') for k,v in means.items()}
        weights={'q':att.q_proj.weight[B.QS].float(),'k':att.k_proj.weight[B.KS].float()}

        def prepare(m,args,kw):state.update(z=kw['hidden_states'],pe=kw['position_embeddings'])

        def capture(key,region):
            def hook(m,args,out):state[key]=out[...,region].contiguous()
            return hook

        def inverse(g,cos,sin):return g*cos-rt.mod.rotate_half(g*sin)

        def rms_grad(g,x,norm):
            x=x.float(); weighted=g*norm.weight.float()
            denom=x.square().mean(-1,keepdim=True)+norm.variance_epsilon
            return (weighted-x*(x*weighted).mean(-1,keepdim=True)/denom)*denom.rsqrt()

        def measure(m,args):
            nonlocal start
            z=state['z']; n,T=z.shape[:2]; chunk=rows[start:start+n]; start+=n
            span=t.zeros((n,5,T),device='cuda',dtype=t.bool)
            for i,row in enumerate(chunk):
                pad=T-len(row['tokens'])
                for j,record in enumerate(row['records']):span[i,j,[pad+x for x in record]]=True
            cos,sin=state['pe']; q0=state['q'][:,-1:]; k0=state['k']
            q=att.q_norm(q0); k=att.k_norm(k0)
            qr=(q*cos[:,-1:]+rt.mod.rotate_half(q)*sin[:,-1:]).float()
            kr=(k*cos+rt.mod.rotate_half(k)*sin).float()
            logits=(qr*kr).sum(-1)*att.scaling
            p=logits[:,None].masked_fill(~span,float('-inf')).softmax(-1)
            diff=p-p.mean(1,keepdim=True)
            q_rot_grad=(diff@kr)*att.scaling
            q_grad=rms_grad(inverse(q_rot_grad,cos[:,-1:].float(),sin[:,-1:].float()),q0,att.q_norm)
            q_terms=(q_grad@weights['q'])*(z[:,-1:].float()-base['q'])
            k_grad=rms_grad(inverse(qr*att.scaling,cos.float(),sin.float()),k0,att.k_norm)
            k_per_token=(k_grad@weights['k'])*(z.float()-base['k'])
            k_terms=diff@k_per_token
            sums[0]+=q_terms.square().sum((0,1)).double().cpu().numpy()
            sums[1]+=k_terms.square().sum((0,1)).double().cpu().numpy()
            counts.append(n)

        att.register_forward_pre_hook(prepare,with_kwargs=True)
        att.q_proj.register_forward_hook(capture('q',B.QS));att.k_proj.register_forward_hook(capture('k',B.KS));att.o_proj.register_forward_pre_hook(measure)
        return result

    Runtime.block=block
    try:B.collect('discovery')
    finally:Runtime.block=original
    assert sum(counts)==len(rows) and np.isfinite(sums).all() and np.all(sums.sum(1)>0)
    B.common.archive('competition-discovery.npz',{'scores':sums,'count':np.array([sum(counts)])})


def select():
    from source_binding_runtime import Runtime
    pre=B.common.checked(); report,blocks=B.read_collection('discovery');assert report['passed']
    path=B.OUT/'competition-discovery.npz'
    with np.load(path) as d:
        assert d['count'][0]==pre['materials']['discovery']['inputs']; scores=d['scores']
    with np.load(REFERENCE/'predictor.npz') as old:
        assert sha(REFERENCE/'predictor.npz')==pre['reference_predictor_sha']
        params={f'{k}_prior_512':old[f'{k}_selected_512'] for k in ('q','k')}
    moment=sum(b['moment'] for b in blocks); n=sum(b['number'] for b in blocks)
    mu=moment[:,0]/n[:,None];rt=Runtime('qwen14'); rng=np.random.default_rng(B.SEED+57); root=f'model.layers.{B.BLOCK}.self_attn.'
    for j,key in enumerate(('q','k')):
        w=rt.weight(root+key+'_proj.weight','cpu')[B.QS if key=='q' else B.KS].contiguous()
        selected=np.argsort(-scores[j],kind='stable');random=rng.permutation(rt.dim)
        params.update({f'mean_{key}':mu[j].astype(np.float32),f'rank_{key}':scores[j],f'weight_{key}':w.view(rt.t.uint16).numpy()})
        for size in B.BUDGETS:params.update({f'{key}_selected_{size}':selected[:size],f'{key}_random_{size}':random[:size]})
    o=rt.weight(root+'o_proj.weight','cpu')[:,B.QS].contiguous();params['weight_o']=o.view(rt.t.uint16).numpy()
    raw=np.concatenate([x['z_final'] for x in blocks]); z=((raw[:,params['q_selected_32']].astype(np.uint32)<<16).view(np.float32)).astype(np.float64)
    center=z.mean(0);spread=np.maximum(z.std(0),1e-6);z=(z-center)/spread
    rules=np.tile(np.array([c[-1] for c in B.CELLS]),B.SIZES['discovery']); coef=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(np.eye(3)[rules]-1/3))
    params.update(probe_mu=center,probe_sd=spread,probe_coef=coef)
    B.common.archive('predictor.npz',params)
    B.common.commit('candidate.json',dict(utc=datetime.now(timezone.utc).isoformat(),prereg_sha=sha(B.OUT/'prereg.json'),discovery_sha=sha(B.OUT/'discovery-collection.json'),
                    ranking_sha=sha(path),predictor_sha=sha(B.OUT/'predictor.npz'),provenance=pre['provenance']))
    print('Sealed competition-based coordinates and equal-budget comparators',flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ('pilot','confirm'): B.collect(stage)
    elif stage in ('freeze','discovery','select','evaluate'):
        {'freeze':freeze,'discovery':discovery,'select':select,'evaluate':B.evaluate}[stage]()
    else:
        import clt_b26_three_cues_intervene as I
        I.freeze() if stage=='freeze-causal' else I.run(stage)
