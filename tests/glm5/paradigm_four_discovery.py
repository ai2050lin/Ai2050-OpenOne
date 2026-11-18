"""Retrospective prediction decomposition and original B30 read/write candidate selection."""

from pathlib import Path
import json

import numpy as np

import paradigm_routes as prior
import paradigm_routes_material as old_material
import paradigm_routes_report as old_report
import record_update_study as ridge
import paradigm_four_material as M
from deploy.storage_policy import checked_dir


OLD=prior.OUT
R,C=prior.R,prior.C
OUTPUT=R.ROOT/'tests/glm5/result/paradigm_four_20261117_v1'
COMMON=('plan.json','candidate-freeze.json','confirm-predictions.npz','confirmation.json','native-parameters.npz','manifest.json')
BLOCK=30


def load(path):
    return json.loads(path.read_bytes())


def bits(x):
    return old_report.unbits(x)


def previous(split,tokenizer,units):
    material=list(old_material.material(tokenizer,split))
    records={k:[] for k in ('target','head_effect','alpha','A','G','U','covariates')}
    manifest=load(OLD/'manifest.json')
    start=0
    for path in sorted(OLD.glob(f'{split}-???.npz')):
        assert C.A.digest(path)==manifest['files'][path.name]['sha256']
        with np.load(path,allow_pickle=False) as z:
            length=len(z['ids'])
            assert list(z['ids'])==[row['id'] for row in material[start:start+length]]
            records['target'].append(bits(z['target']))
            records['head_effect'].append(z['head_effect'][:,BLOCK])
            records['alpha'].append(z['alpha'][:,BLOCK])
            activation=z['A']
            records['covariates'].append(bits(np.stack([activation[:,l,j] for l,j in units],1)))
            records['A'].append(bits(activation[:,BLOCK]))
            del activation
            for key in ('G','U'):
                records[key].append(bits(z[key][:,BLOCK]))
        start+=length
        print('Read',split,start,'/',len(material),flush=True)
    assert start==len(material)
    return material,{name:np.concatenate(parts) for name,parts in records.items()}


def design_residual(x,y,penalty=2.):
    # The input includes an intercept. Scale is learned on the available cohort.
    x=x.astype(np.float64)
    penalty_matrix=np.eye(x.shape[1])*penalty
    penalty_matrix[0,0]=0
    beta=np.linalg.solve(x.T@x+penalty_matrix,x.T@y)
    return (y-x@beta).astype(np.float32)


def cohort_fit(text,test,target):
    fit=ridge.linear(text,target,12.)
    return ridge.evaluate(fit,test)


def family_delta(errors):
    return old_report.family_summary(np.asarray(errors).reshape(-1,8).mean(1))


def explain_random(tokenizer,optical):
    old_units=optical['units'];old_random=optical['random_units']
    rng=np.random.default_rng(M.SEED)
    random_sets=[old_random]
    for _ in range(24):
        occupied={tuple(u) for u in old_units}
        draw=[]
        for layer,_ in old_units:
            unit=int(rng.integers(17408))
            while (layer,unit) in occupied:
                unit=int(rng.integers(17408))
            occupied.add((layer,unit));draw.append([layer,unit])
        random_sets.append(draw)
    flat=[*old_units,*(coord for group in random_sets for coord in group)]
    discovery,d=previous('discovery',tokenizer,flat)
    confirm,c=previous('confirm',tokenizer,flat)
    assert optical['head'][0]==BLOCK
    head=optical['head'][1]
    x=old_report.text_features(discovery)
    y=old_report.text_features(confirm)
    # Use the original text feature definition and identical head covariate in both arms.
    with_head=np.c_[x,d['head_effect'][:,head]]
    test_head=np.c_[y,c['head_effect'][:,head]]
    y_train=d['target'];y_test=c['target']
    errors={}
    for name,train,test in [('text',x,y),('head_only',with_head,test_head)]:
        estimate=cohort_fit(train,test,y_train)
        errors[name]=((estimate-y_test)**2).mean(1)
    for no in range(1+len(random_sets)):
        span=slice(no*32,(no+1)*32)
        estimate=cohort_fit(np.c_[with_head,d['covariates'][:,span]],
                            np.c_[test_head,c['covariates'][:,span]],y_train)
        name='selected' if no==0 else f'random_{no-1:02}'
        errors[name]=((estimate-y_test)**2).mean(1)
    means={name:float(value.mean()) for name,value in errors.items()}
    activations=d['covariates'][:,:64]
    cells=np.eye(8,dtype=np.float32)[np.arange(len(discovery))%8]
    cell_residual=design_residual(cells,activations)
    explained=1-cell_residual.var(0)/np.maximum(activations.var(0),1e-12)
    scaled=(activations-activations.mean(0))/np.maximum(activations.std(0),1e-8)
    def rank(v):
        spectrum=np.linalg.svd(v,compute_uv=False)**2
        p=spectrum/np.maximum(spectrum.sum(),1e-20)
        return float(1/np.maximum((p*p).sum(),1e-20))
    with np.load(OLD/'confirm-predictions.npz',allow_pickle=False) as values:
        delta=(values['native'][:,0]-values['random'][:,0]).reshape(-1,8)
        recorded=dict(all=family_delta(delta.ravel()),cells=delta.mean(0).tolist(),
                      mlp_selected_KL=float(values['native'][:,0].mean()),random_KL=float(values['random'][:,0].mean()))
    explained_groups=[float(v.mean()) for v in (explained[:32],explained[32:])]
    diagnostic=dict(scope='Phase 2964 already-exposed confirmation replay only. Exact recorded output KL, and 25 independently frozen layer-count-matched sets evaluated by final normalized-state MSE. No new blind claim.',
                    recorded=recorded,hidden_mse=means,draws_better_than_selected=sum(means[f'random_{i:02}']<means['selected'] for i in range(25)),
                    selected_minus_first_random_mse=family_delta(errors['selected']-errors['random_00']),
                    explained_by_factor_cells=dict(selected=explained_groups[0],random=explained_groups[1]),
                    activation_effective_rank=dict(selected=rank(scaled[:,:32]),random=rank(scaled[:,32:])),
                    arm_units=random_sets)
    return diagnostic,discovery,d,x


def original_weights(heads):
    import torch
    from safetensors import safe_open

    model=R.MODEL
    index=load(model/'model.safetensors.index.json')['weight_map']
    prefix=f'model.layers.{BLOCK}.'
    def get(key):
        name=prefix+key+'.weight'
        with safe_open(model/index[name],framework='pt',device='cpu') as file:
            return file.get_tensor(name).float().numpy()
    output=get('self_attn.o_proj')
    gamma=get('post_attention_layernorm')
    gate=get('mlp.gate_proj')
    up=get('mlp.up_proj')
    # Physical O columns and original MLP rows. This is the numerator of the
    # RMSNorm path; original nonlinear G/U are recomputed under native edits.
    columns=np.stack([output[:,h*128:(h+1)*128] for h in heads])
    connected=[]
    for matrix in (gate,up):
        state=matrix*gamma[None,:]
        scale=np.maximum(np.linalg.norm(state,axis=1),1e-8)
        projected=np.stack([np.linalg.norm(state@column,axis=1)/scale for column in columns])
        connected.append(projected)
    return (*connected,gate,up)


def rank_candidates(optical,discovery,d,text):
    from scipy.stats import rankdata

    request=np.array([row['regime'][0] for row in discovery],bool)
    reference=np.array([row['regime'][1] for row in discovery],bool)
    mass=d['alpha']
    ratio=np.log(np.maximum(mass[:,:,0],1e-8)/np.maximum(mass[:,:,1],1e-8))
    source_change=np.abs(ratio[request].mean(0)-ratio[~request].mean(0))
    reference_change=np.abs(ratio[reference].mean(0)-ratio[~reference].mean(0))
    old_head=optical['head'][1]
    peers=np.asarray([h for h in range(40) if h//5!=old_head//5])
    peer_score=(rankdata(source_change)+rankdata(reference_change))
    other=int(peers[np.argmax(peer_score[peers])])
    heads=[old_head, other]
    gate_link,up_link,gate_weight,up_weight=original_weights(heads)
    conditional=[]
    regimes=np.arange(len(discovery))%8
    for key in ('G','U'):
        values=d[key]
        numerator=np.std(np.stack([values[regimes==j].mean(0) for j in range(8)]),axis=0)
        conditional.append(numerator/np.maximum(values.std(0),1e-6))
    g,u=conditional
    rng=np.random.default_rng(M.SEED+4)
    projection=rng.choice(np.array([-1.,1.],np.float32),size=(5120,32))/np.sqrt(5120)
    output_residual=design_residual(text,d['target']@projection)
    ar=design_residual(text,d['A'])
    pairs=np.arange(len(ar))//8
    half=(pairs%2)==0
    correlations=[]
    for mask in (half,~half):
        x=ar[mask];y=output_residual[mask]
        denom=np.maximum(np.linalg.norm(x,axis=0)[:,None]*np.linalg.norm(y,axis=0)[None,:],1e-8)
        correlations.append(x.T@y/denom)
    stable=np.maximum(np.sum(correlations[0]*correlations[1],axis=1),0)
    links=np.maximum(gate_link.mean(0),up_link.mean(0))
    score=rankdata(np.maximum(g,u))*rankdata(stable)*rankdata(links)
    chosen=np.argsort(-score,kind='stable')[:32]
    random=np.array(rng.choice(np.setdiff1d(np.arange(17408),chosen),32,replace=False),int)
    record=dict(block=BLOCK,heads=heads,legacy_head=old_head,
                head_route_effect=[dict(head=h,request=float(source_change[h]),reference=float(reference_change[h])) for h in heads],
                selected=chosen.tolist(),random=random.tolist(),
                rank_definition='Rank product of G/U conditional fraction; split-half normalized residual correlation; exact original Wgate/up diag(gamma) WO connection numerator norm, selected B30. All original Phase 2964 discovery only.',
                cards=[dict(unit=int(j),gate_fraction=float(g[j]),up_fraction=float(u[j]),gate_links=gate_link[:,j].tolist(),
                            up_links=up_link[:,j].tolist(),conditional_residual_alignment=float(stable[j]),
                            primary_gate_coordinate=int(abs(gate_weight[j]).argmax()),primary_up_coordinate=int(abs(up_weight[j]).argmax()))
                       for j in chosen])
    return record,dict(gate_fraction=g,up_fraction=u,gate_link=gate_link,up_link=up_link,residual_reliability=stable,rank_score=score)


def run():
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    checked_dir(OUTPUT,384*2**20,operation='范式全场复用与新实验预算')
    if (OUTPUT/'candidate.json').exists():
        raise FileExistsError('Existing retrospective candidate freeze')
    R.OUT=OUTPUT
    prior.OUT=OUTPUT
    tokenizer=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    optical=load(OLD/'candidate-freeze.json')
    diagnostic,discovery,data,design=explain_random(tokenizer,optical)
    R.write('retrospective.json',diagnostic)
    candidate,arrays=rank_candidates(optical,discovery,data,design)
    rank_sha=prior.write_np('all-native-ranking.npz',arrays)
    inputs={name:C.A.digest(OLD/name) for name in COMMON}
    discovered={path.name:C.A.digest(path) for path in OLD.glob('discovery-???.npz')}
    path=Path(__file__).relative_to(R.ROOT).as_posix()
    R.write('candidate.json',dict(at=C.stamp(),sources=inputs,discovery=discovered,
            algorithm={p:C.A.digest(R.ROOT/p) for p in (path,'tests/glm5/paradigm_four_material.py')},
            ranking=rank_sha,retrospective=C.A.digest(OUTPUT/'retrospective.json'),**candidate))
    print('Frozen native candidates',candidate['heads'],candidate['selected'][:8],flush=True)


if __name__=='__main__':
    run()
