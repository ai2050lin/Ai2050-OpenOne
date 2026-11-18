"""Freeze, measure behavior, and run same-shape Qwen3-14B path counterfactuals."""

import argparse
from itertools import batched, combinations
import math
import time

import numpy as np

import paradigm_path_calibration as A
import paradigm_path_core as P
import paradigm_path_material as M
import paradigm_four_run as F
import paradigm_four_report as V
from deploy.storage_policy import check_write
from tools.power_analysis import check_phase


R,C,OUT=A.R,A.C,A.OUT
SLAB=128
NATIVE=('zero','archive','request','joint','source_matched','control_matched',
        'gate_restore','up_restore','both_restore','random_restore','both_matched','random_matched')
COMMON=('sham','restored')
ARMS=(*COMMON,*(f'{h}/{op}' for h in P.HEADS for op in NATIVE))
PAIR_LABELS=tuple(combinations((8,4,2,1),2))
DOSES=('source','restore')


def digest(path):
    return C.A.digest(path)


def hashes():
    files=F.hash_sources()
    files.update({f.as_posix():digest(R.ROOT/f) for f in
                  [p.relative_to(R.ROOT) for p in OUT.parents[1].glob('paradigm_path_*.py')]})
    return files


def bind_inputs():
    return {name:digest(path) for name,path in {
        'calibration':OUT/'calibration.json','calibration_data':OUT/'calibration.npz',
        'parent_manifest':A.PARENT/'manifest.json','candidate':A.PARENT/'candidate.json',
        'native_parameters':A.PARENT/'native-parameters.npz'}.items()}


def freeze():
    from transformers import AutoTokenizer

    if (OUT/'plan.json').exists():
        raise FileExistsError('Already frozen')
    calibration=R.read('calibration.json')
    snapshot=C.Prior.identity()
    assert calibration['passed'] and calibration['snapshot']==snapshot
    assert calibration['artifacts']==digest(OUT/'calibration.npz')
    assert calibration['source']==digest(A.PARENT/'manifest.json')
    candidate=read_candidate()
    assert candidate['block']==P.LAYER and candidate['heads']==list(P.HEADS)
    assert len(candidate['selected'])==len(candidate['random'])==32
    assert not set(candidate['selected'])&set(candidate['random'])
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    power=check_phase(.14,128,.20,.01)
    assert power['pass']
    print('Frozen',R.write('plan.json',dict(
        at=C.stamp(),run=M.RUN,provenance=R.metadata(),snapshot=snapshot,inputs=bind_inputs(),
        code=hashes(),versions=R.versions(),material=M.identities(tok),power=power,
        anchors=F.prior_material.alphabet(tok),arms=ARMS,heads=P.HEADS,control=P.CONTROL,
        dose=dict(min_positive=P.ROW_DOSE,relative_gap=P.TOL,min_cosine=P.COS_MIN,
                  minimum_coverage=A.TECHNICAL_MIN,grid=P.GRID),
        gates=dict(behavior=.75,behavior_by_nuisance='all 8 pilot nuisance strata for each factor',
          repair='all fresh onset full-vocabulary logits equal bit for bit in sham and both restored-head replays',
          fusion='primary 1011 and 1111 in natural archives; signed KL improvement >=10% and 99% family CI lower >0; report all paired archive swaps separately', 
          competition='for each head and target in conflict: format pilot gate, exact repair, >=0.10 signed mean for archive removal and negative of request removal, both 99% lower CI >0, also >=98% dose coverage and signed matched-source vs control contrast with lower CI >0',
          mediation='for each head and target/archive state: only assert direction-specific rescue with >=98% dose match, signed selected restoration improves toward sham and improves over matched random, both lower 99% CI >0. Record all native swaps and sign reversals; original native G/U swaps have variable dose.'),
        design=dict(estimand='free onset full-vocabulary and print minus logsumexp(prose anchors). The other three factors are measured from prefilled condition probes. No change to tokens in intervention.',
          cohorts='16 pilot, 128 new confirm, 128 untouched model lockbox, with all 16 observed conditions. First 32 confirmation families crossed with two historical archive styles for the entire design. First 64 crossed at only 1011 and 1111 for causal path.',
          fusion='Predict each target from base, its singles, then its doubles. Store KL, argmax matches, and single-omitted pair ablations. This is conditional generalization across combinations within a new family, not prediction from text alone.',
          attention='Full original BF16 shape. At B30 and onset E, zero the whole H25 or H18 write, or independently remove archive or request V (or both), fixed QK. Install only the named head output, despite shared KV. Retain full native restore and a signed matched source contrast versus B30/H20 archive, which is a single same-block comparator.',
          neurons='On archive cut: replace only selected original G, U, or both; random 32 G/U reference; separately scale measured native displacement directions to a common realized displacement in BF16 after B30 MLP. FP32 algebra direction is retained and used only for a zero native displacement.',
          statistics='99% family bootstrap. Exploratory contrasts not used to choose new conditions after reading. Signed contrasts use desired style sign. Repeat with archive-aligned stratum. Use mask-defined eligible views for matched contrasts; retain all failures and raw data.',
          compute='BF16 40-block one-GPU sequential engine. D-only space checks every slab. Save two complete natural confirm families at onset and conditioned count in all 41 boundaries. No full vocabulary logits on disk.'))),flush=True)


def read_candidate():
    import json
    return json.loads((A.PARENT/'candidate.json').read_bytes())


def panels(model,group,field):
    result=[]
    for i,rows in enumerate(batched(group,8)):
        panel=C.packet(model,rows)
        panel['q']=model.t.tensor([r['query'] for r in rows],device='cuda')
        if i<field:
            panel['field']=[]
        result.append(panel)
    return result


def factor_prediction(model,logits):
    t=model.t
    z=t.as_tensor(np.stack(logits),device='cuda').view(t.bfloat16).float()
    base=z[0]
    scores=[]
    predicted=[]
    for target in M.TARGET:
        singles=[bit for bit in (8,4,2,1) if target&bit]
        add=base+(z[singles]-base).sum(0)
        terms={pair:z[sum(pair)]-z[pair[0]]-z[pair[1]]+base for pair in PAIR_LABELS
               if all(target&bit for bit in pair)}
        pair=add+t.stack(list(terms.values())).sum(0)
        y=z[target].log_softmax(-1)
        guesses=t.stack([base,add,pair,*(pair-terms[key] if key in terms else pair for key in PAIR_LABELS)])
        kl=(y.exp()*(y-guesses.log_softmax(-1))).sum(-1)
        tops=guesses.argmax(-1)
        scores.append(t.stack((kl,(tops==y.argmax()).float()),-1).cpu().numpy())
        predicted.append(t.cat((y.argmax().reshape(1),tops)).cpu().numpy())
    return np.stack(scores),np.stack(predicted)


def native(model,block,p,x,q,k,v,o,candidate):
    t=model.t
    pos=p['q'][:,0]
    clean=P.extract(model,block,x,o,pos)
    j=clean['j']
    control=P.source_cut(model,block,p,x,q,k,v,o,P.CONTROL,'archive')
    restored=clean['output'][j,pos]
    branches=[restored]
    doses=[];values=[];mass=[];whole=[]
    groups=candidate['selected']+candidate['random']
    values.append(P.selected_gua(model,clean,groups))
    for h in P.HEADS:
        head=o.clone()
        head[j,pos,h]=0
        zero=P.extract(model,block,x,head,pos)
        archive=P.source_cut(model,block,p,x,q,k,v,o,h,'archive')
        request=P.source_cut(model,block,p,x,q,k,v,o,h,'request')
        joint=P.source_cut(model,block,p,x,q,k,v,o,h,'archive+request')
        # Restore this head's unedited vector in the original position and shape.
        head[j,pos,h]=o[j,pos,h]
        full=P.extract(model,block,x,head,pos)['output'][j,pos]
        assert t.equal(full,restored)
        whole.append(full)
        first,second,source=P.realized_match(clean['after'][j,pos],archive['after'][j,pos],control['after'][j,pos],t,
                                             (archive['direction'],control['direction']))
        restores=[P.neuron_restore(model,block,clean,archive,candidate['selected'],mode)
                  for mode in ('gate','up','both')]
        single=[state[j,pos] for state,_ in restores]
        random,rd=P.neuron_restore(model,block,clean,archive,candidate['random'],'both')
        random=random[j,pos]
        matched,matched_random,rescue=P.realized_match(archive['output'][j,pos],single[-1],random,t,(restores[-1][1],rd))
        branches.extend([state['output'][j,pos] for state in (zero,archive,request,joint)])
        branches.extend((P.from_attention(model,block,clean,first),P.from_attention(model,block,clean,second),
                         *single,random,matched,matched_random))
        doses.append(np.stack((source,rescue),1))
        values.extend(P.selected_gua(model,state,groups) for state in (zero,archive,request,joint))
        mass.append(t.stack([(state['after'][j,pos].float()-clean['after'][j,pos].float()).norm(dim=1)
                             for state in (zero,archive,request,joint)],1).cpu().numpy())
    branches.insert(1,whole[0])
    p['branches']=t.stack(branches).cpu()
    p['doses']=np.stack(doses,1)
    p['gua']=np.stack(values,1)
    p['attention_norm']=np.stack(mass,1)
    q0=q[j,:,pos].unsqueeze(2)
    visible=p['mask'].cuda()[j,:,pos].unsqueeze(2)
    regions=F.span_masks(model,p)
    reads=model.read(q0,k,visible)[:,[*P.HEADS,P.CONTROL]]
    p['attention_mass']=(reads[:,:,None]*regions[:,None]).sum(-1).cpu().numpy()
    assert len(branches)==len(ARMS)


def propagate(model,part,group,label,started,total,candidate=None,field=0):
    t=model.t
    collection=panels(model,group,field)
    C.N.progress(label,part*40,total,started)
    for layer in range(40):
        with model.block(layer) as block:
            for p in collection:
                x=p['x'].cuda()
                mask=p['mask'].cuda()
                angles=[a.cuda() for a in p['angles']]
                if 'field' in p:
                    j=t.arange(len(x),device='cuda')
                    p['field'].append(F.base.bit_pattern(x[j[:,None],p['q'][:,[0,3]]]))
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,mask)
                clean=model.residual(block,x,o)
                if candidate and layer==P.LAYER:
                    native(model,block,p,x,q,k,v,o,candidate)
                elif candidate and layer>P.LAYER:
                    p['branches']=t.stack([P.same_shape_next(model,block,x,b,p,angles)
                                           for b in p['branches'].cuda()]).cpu()
                if candidate and layer>=P.LAYER:
                    j=t.arange(len(x),device='cuda')
                    assert t.equal(p['branches'][0],clean[j,p['q'][:,0]].cpu()),layer
                    assert t.equal(p['branches'][0],p['branches'][1]),layer
                p['x']=clean.cpu()
        if (layer+1)%5==0:
            C.N.progress(label,part*40+layer+1,total,started)
    return collection


def scores(model,collection,setup,causal):
    t=model.t
    raw=[];pool=[];fusion=[];fusion_top=[]
    for p in collection:
        x=p['x'].cuda()
        logits=P.full_logits(model,p,x)
        row=dict(ids=np.array([r['id'] for r in p['rows']]),
                 family=np.array([r['family'] for r in p['rows']]),
                 cell=np.array([sum(b<<i for i,b in enumerate(reversed(r['dims']))) for r in p['rows']]),
                 archive=np.array([r['archive'] for r in p['rows']]),
                 margin=F.probe(model,p,logits,setup),top=logits.argmax(-1).cpu().numpy())
        j=t.arange(len(x),device='cuda')
        if 'field' in p:
            row['field']=np.stack((*p['field'],F.base.bit_pattern(x[j[:,None],p['q'][:,[0,3]]])),1)
            row['field_ids']=row['ids']
        if causal:
            edit_margin=[];edit_top=[];edit_kl=[]
            q=p['q'][:,0]
            baseline=logits[:,0].log_softmax(-1)
            for k,b in enumerate(p['branches'].cuda()):
                active=x.clone();active[j,q]=b
                edited=P.full_logits(model,p,active)[:,0]
                if k<len(COMMON):
                    assert t.equal(edited,logits[:,0])
                edit_margin.append(F.opening_scores(model,edited,setup[:2]).cpu().numpy())
                edit_top.append(edited.argmax(-1).cpu().numpy())
                edit_kl.append((baseline.exp()*(baseline-edited.log_softmax(-1))).sum(-1).cpu().numpy())
            row.update(edit=np.stack(edit_margin,1),edit_top=np.stack(edit_top,1),edit_kl=np.stack(edit_kl,1),
                       dose=p['doses'],gua=p['gua'],attention_norm=p['attention_norm'],attention_mass=p['attention_mass'])
        else:
            pool.extend(F.base.bit_pattern(logits[:,0].to(t.bfloat16)))
            if len(pool)==16:
                ms,ts=factor_prediction(model,pool)
                fusion.append(ms);fusion_top.append(ts)
                pool=[]
        raw.append(row)
    assert not pool
    arrays={key:np.concatenate([row[key] for row in raw if key in row]) for key in set().union(*raw)}
    if not causal:
        f=arrays['family'].reshape(-1,16);a=arrays['archive'].reshape(-1,16)
        assert (f==f[:,:1]).all() and (a==a[:,:1]).all()
        assert (arrays['cell'].reshape(-1,16)==np.arange(16)).all()
        arrays.update(fusion=np.stack(fusion),fusion_top=np.stack(fusion_top))
    return arrays


def measure(model,rows,label,plan,setup,candidate=None):
    started=time.monotonic()
    total=40*math.ceil(len(rows)/SLAB)
    names=[]
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'{label}-{part:02}.npz'
        receipt=f'{label}-{part:02}.json'
        plan_sha=digest(OUT/'plan.json')
        if (OUT/receipt).exists():
            prior=R.read(receipt)
            assert prior['sha']==digest(OUT/name) and prior['plan']==plan_sha
        else:
            check_write(OUT/name,36*2**20,operation='Qwen B30 逐批磁盘预检')
            field=8 if label=='confirm' and part==0 else 0
            collection=propagate(model,part,group,label,started,total,candidate,field)
            arrays=scores(model,collection,setup,candidate is not None)
            sha=F.base.write_np(name,arrays)
            R.write(receipt,dict(sha=sha,plan=plan_sha,views=len(group),at=C.stamp()))
        names.append(name)
    R.write(label+'-files.json',dict(files=names,plan=digest(OUT/'plan.json')))
    return names


def anchors(model,plan):
    t=model.t
    words=model.tok.batch_decode([[i] for i in range(model.model.lm_head.weight.shape[0])])
    han=t.tensor([any('\u3400'<=c<='\u9fff' for c in word) for word in words],device='cuda')
    latin=t.tensor([word.lstrip()[:1].isascii() and word.lstrip()[:1].isalpha() for word in words],device='cuda')
    a=plan['anchors']
    return a['print'],sorted(set(a.values())-{a['print']}),han,latin


def load(label):
    prior=R.read(label+'-files.json')
    assert prior['plan']==digest(OUT/'plan.json')
    V.OUT=OUT
    return V.data_load(prior['files'])


def run():
    import torch

    plan=R.read('plan.json')
    assert plan['code']==hashes() and plan['inputs']==bind_inputs() and plan['versions']==R.versions()
    candidate=read_candidate()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('path-load',0,2,start)
        assert plan['snapshot']==C.Prior.identity()
        model=C.N.Engine()
        C.N.progress('path-load',2,2,start)
        try:
            setup=anchors(model,plan)
            pilot,_=M.cohort(model.tok,'pilot')
            assert M.digest([pilot,[]])==plan['material']['pilot']['identity']
            measure(model,pilot,'pilot',plan,setup)
            gate=V.behavior(load('pilot'),pilot,model.tok)
            R.write('behavior.json',gate)
            print('Pilot gates',{key:value['minimum_stratum'] for key,value in gate.items()},flush=True)
            rows,causal=M.cohort(model.tok,'confirm')
            assert M.digest([rows,causal])==plan['material']['confirm']['identity']
            if gate['format']['passed']:
                measure(model,causal,'native',plan,setup,candidate)
            if all(value['passed'] for value in gate.values()):
                measure(model,rows,'confirm',plan,setup)
        finally:
            model.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','run'))
    args=cli.parse_args()
    A.setup()
    (freeze if args.action=='freeze' else run)()
