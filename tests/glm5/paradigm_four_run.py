"""Held-out factorial accounting and native source/MLP intervention on Qwen3-14B."""

from itertools import batched, combinations
from pathlib import Path
import argparse
import math
import time

import numpy as np

import kv_chain_native as K
import paradigm_four_discovery as D
import paradigm_four_material as M
import paradigm_routes as base
import paradigm_routes_material as prior_material
from deploy.storage_policy import check_write,checked_dir,StorageBlocked
from tools.power_analysis import check_phase


R,C,OUT=base.R,base.C,D.OUTPUT
TEMP=R.ROOT/'tests/glm5_temp'/OUT.name
ARMS=('sham','archive_support_off','archive_value_off','request_value_off','joint_off','restore_all',
      'joint_selected_restore','joint_random_restore')
FACTORS=('language','format','logical_source','grammar')
MASKS=[[i for i,n in enumerate(cell) if n] for cell in M.LEVELS]
COMBINED=[i for i,subset in enumerate(MASKS) if len(subset)>=3]
BATCH=8
SLAB=256
PEAK=384*2**20


def setup():
    R.OUT,R.TEMP=OUT,TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH=M.RUN,M.SEED,M.WIDTH
    R.BATCH,R.SLAB=BATCH,SLAB
    base.OUT=OUT
    R.configure(PEAK)


def hash_sources():
    files=set(base.BASE_SOURCES())
    files.update(f'tests/glm5/paradigm_four_{name}.py' for name in ('material','discovery','run','report'))
    return {f:C.A.digest(R.ROOT/f) for f in sorted(files)}


def freeze():
    from transformers import AutoTokenizer

    if (OUT/'plan.json').exists():raise FileExistsError('Frozen study already exists')
    checked_dir(OUT,PEAK,operation='四范式冻结与采集预算')
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    material={}
    for split in M.SIZES:
        rows=M.validate(tok,split)
        material[split]=dict(views=len(rows),families=M.SIZES[split],sha=M.digest(rows),nouns=sorted({r['noun'] for r in rows}))
    assert not (set(material['pilot']['nouns'])&set(material['confirm']['nouns']))
    assert not (set(material['lockbox']['nouns'])&set(material['confirm']['nouns']))
    power=check_phase(.14,128,.20,.01)
    assert power['pass']
    plan=dict(at=C.stamp(),run=M.RUN,provenance=R.metadata(),snapshot=C.Prior.identity(),sources=hash_sources(),
              candidate=C.A.digest(OUT/'candidate.json'),material=material,version=R.versions(),power=power,
              axes=FACTORS,anchors=prior_material.alphabet(tok),arms=ARMS,
              design=dict(
                  cohort='16 development families, 128 new confirmation families and 128 unopened lockbox families. Full 16-cell design. New nouns and a distinct frame. Archive style balanced across independent noun/count families.',
                  scope='Paper or bare Python print, Chinese or English, initial vs corrected count selection, round vs square delimiter syntax. The latter three measurement slots use predetermined continuation prefixes where needed and do not certify full autonomous text.',
                  behavior='At each fixed combination of other factors, compare 16 matched pilot pairs for positive native margin changes on each of four factors; require >=.75 in every nuisance stratum for that factor. Report all absolute outcomes as separate diagnostics.',
                  natural='At shared free onset save full-vocabulary argmax/first contrast, six original QK source groups for frozen heads and all sixteen factor conditions. Also read language at Python message boundary, delimiter before bracket, count after supplied bracket. Preserve all 41 hidden boundaries at onset and count for first two complete confirm families.',
                  prediction='On each new family prospectively compute z_add from only all-zero and singleton conditions, and z_pair after also incorporating double conditions. Score full-vocabulary KL on triple and fourfold conditions. Compare no-factor base, additive, pairwise. Require pairwise improvement >=10% vs additive and positive 99% family-bootstrap lower bound. No predicted cell output enters its features.',
                  causal='Independently of paired behavior flags, assert only physical local effects at free onset. Frozen B30 heads and selected plus same-block random neurons. On first 64 confirmation families address-mask archive, clear archive Value, request Value and both; restore original head outputs or restore selected/random original MLP activation within joint block. Recompute live query propagation. For positive functional format competition require pilot format-behavior pass, >=60 exact-grid sham top1-aligned families and positive confidence bound with >=.10 mean for both archive removal helping requested style and request removal hurting it in conflict. No causal extension to grammar/count readouts.',
                  dose='Native full-source edits recorded with realized projected norms and zero-fold variants as natural counterfactuals. Selected/random MLP rescue dose errors reported, matched if P95 <=.03; superiority of selected rescue conditional on dose match. Full native sham restoral exact. Onset/conditioned query results and failure criteria frozen.',
                  resources='Stream original 40 BF16 blocks through CUDA one model. At most 384 MiB added peak. Store full logits only while scoring each slab, physical effects and auditable trajectory slices on D. Slab work receipts and checkpoint hashes.'),
              runtime=dict(width=M.WIDTH,batch=BATCH,slab=SLAB,peak_bytes=PEAK))
    print('Frozen',R.write('plan.json',plan),flush=True)


def verify():
    plan=R.read('plan.json')
    assert plan['sources']==hash_sources() and plan['version']==R.versions()
    assert plan['candidate']==C.A.digest(OUT/'candidate.json')
    return plan


def span_masks(model,panel):
    t=model.t
    if 'regions' not in panel:
        valid=t.zeros((len(panel['rows']),len(M.REGIONS),M.WIDTH),dtype=t.bool,device='cuda')
        for i,row in enumerate(panel['rows']):
            for j,key in enumerate(M.REGIONS):
                valid[i,j,row['spans'][key]]=True
        panel['regions']=valid
    return panel['regions']


def probe(model,panel,logits,setup_weights):
    t=model.t
    j=t.arange(len(panel['rows']),device='cuda')
    print_id,formal,han,latin=setup_weights
    onset=logits[:,0]
    format_=onset[:,print_id]-onset[:,formal].logsumexp(-1)
    message=logits[:,1]
    language=message[:,han].logsumexp(-1)-message[:,latin].logsumexp(-1)
    delim=t.tensor([r['delimiters'] for r in panel['rows']],device='cuda')
    grammar=logits[j,2,delim[:,1]]-logits[j,2,delim[:,0]]
    count=t.tensor([r['number'] for r in panel['rows']],device='cuda')
    logic_gold=logits[j,3,count[:,0]]-logits[j,3,count[:,1]]
    logic=logic_gold*t.tensor([1 if r['dims'][2] else -1 for r in panel['rows']],device='cuda')
    return t.stack((language,format_,logic,grammar),-1).cpu().numpy()


def conditional_write(model,block,index,panel,x,q,k,v,o,mask,candidate,query):
    t=model.t
    j=t.arange(len(x),device='cuda');qpos=query[:,0];end=query[:,-1]
    if panel['keep_field']:
        panel.setdefault('field',[]).append(base.bit_pattern(x[j[:,None],query[:,[0,3]]]))
    if index!=candidate['block']:
        return model.residual(block,x,o)
    heads=candidate['heads']
    outs=block.self_attn.o_proj(o.reshape(*x.shape[:-1],model.width).contiguous())
    after=x+outs
    normalized=block.post_attention_layernorm(after)
    g=block.mlp.gate_proj(normalized);u=block.mlp.up_proj(normalized)
    a=t.nn.functional.silu(g)*u
    units=candidate['selected']+candidate['random']
    rows=j[:,None]
    measured=[base.bit_pattern(value[rows,query[:,[0,3]]][:,:,units]) for value in (g,u,a)]
    panel['gua']=np.stack(measured,-1)
    groups=span_masks(model,panel)
    info=[]
    for pos in (qpos,end):
        query_q=q[j,:,pos].unsqueeze(2)
        visible=mask[j,:,pos].unsqueeze(2)
        mass=model.read(query_q,k,visible)[:,heads]
        info.append((mass[:,:,None,:]*groups[:,None]).sum(-1).cpu().numpy())
    panel['mass']=np.stack(info,1)
    return after+block.mlp.down_proj(a)


def native_step(model,block,index,panel,x,k,v,mask,angles,query,candidate):
    t=model.t
    j=t.arange(len(x),device='cuda')
    pos=query[:,0];b=len(j);arm_count=len(ARMS)
    live=x[j,pos].expand(arm_count,-1,-1).clone() if index==candidate['block'] else panel['live'].cuda()
    query_angle=[p[j,pos,None].repeat(arm_count,1,1) for p in angles]
    q,k_current,v_current=model.qkv(block,live.reshape(arm_count*b,1,model.width),query_angle)
    keys=k.repeat(arm_count,1,1,1);values=v.repeat(arm_count,1,1,1)
    indices=t.arange(arm_count*b,device='cuda');positions=pos.repeat(arm_count)
    keys[indices,:,positions]=k_current[:,:,0];values[indices,:,positions]=v_current[:,:,0]
    visible=mask[j,:,pos].unsqueeze(2)
    wrote=model.attention(block,q,keys,values,visible.repeat(arm_count,1,1,1)).reshape(arm_count,b,40,128)
    if index==candidate['block']:
        original=wrote[0].clone()
        span=span_masks(model,panel)
        archive=span[:,M.REGIONS.index('archive')]
        request=span[:,M.REGIONS.index('format')]
        for h in candidate['heads']:
            for arm,target,mode in ((1,archive,'address'),(2,archive,'value'),(3,request,'value'),(4,archive|request,'value')):
                change=K.raw_change(model,block,h,q[:b],keys[:b],values[:b],visible,target,mode)
                wrote[arm,:,h]=(original[:,h].float()+change).to(wrote.dtype)
        wrote[5]=original
        wrote[6:8]=wrote[4].clone()
        projected=block.self_attn.o_proj(wrote.reshape(arm_count,b,model.width).contiguous())
        panel['head_doses']=(projected.float()-projected[0].float()).norm(dim=-1).T.cpu().numpy()
        after=live+projected
        normalized=block.post_attention_layernorm(after)
        g=block.mlp.gate_proj(normalized);u=block.mlp.up_proj(normalized)
        a=t.nn.functional.silu(g)*u
        units=candidate['selected']+candidate['random']
        panel['native_gua']=t.stack((g[:,:,units],u[:,:,units],a[:,:,units]),-1).transpose(0,1).to(t.float32).cpu().numpy()
        for arm,selected in ((6,candidate['selected']),(7,candidate['random'])):
            a[arm,:,selected]=a[0,:,selected]
        down=block.mlp.down_proj(a)
        panel['mlp_doses']=(down[6:8].float()-down[4].float()).norm(dim=-1).T.cpu().numpy()
        panel['live']=(after+down).cpu()
    else:
        panel['live']=model.residual(block,live,wrote).cpu()


def opening_scores(model, logit, anchors):
    return logit[...,anchors[0]]-logit[...,anchors[1]].logsumexp(-1)


def fusion_stats(model,logits):
    t=model.t
    z=t.as_tensor(np.stack(logits),device='cuda').float()
    assert z.shape[0]==16
    baseline=z[0]
    scores=[]
    for i in COMBINED:
        singleton=[8>>j for j in MASKS[i]]
        addon=z[singleton]-baseline
        additive=baseline+addon.sum(0)
        pair=t.stack([z[a+b]-z[a]-z[b]+baseline for a,b in combinations(singleton,2)])
        pairwise=additive+pair.sum(0)
        p=z[i].log_softmax(0)
        guesses=t.stack((baseline,additive,pairwise))
        kl=(p.exp()*(p-guesses.log_softmax(-1))).sum(-1)
        top=(guesses.argmax(-1)==z[i].argmax()).float()
        scores.append(t.stack((kl,top),-1).cpu().numpy())
    return np.stack(scores)


def measure(model,split,rows,plan,candidate):
    t=model.t
    n_vocab=model.model.lm_head.weight.shape[0]
    vocab=model.tok.batch_decode([[i] for i in range(n_vocab)])
    han=t.tensor([any('\u3400'<=c<='\u9fff' for c in w) for w in vocab],device='cuda')
    latin=t.tensor([w.lstrip()[:1].isascii() and w.lstrip()[:1].isalpha() for w in vocab],device='cuda')
    anchors=plan['anchors']
    setup_weights=anchors['print'],sorted(set(anchors.values())-{anchors['print']}),han,latin
    started=time.monotonic()
    results=[]
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'{split}-{part:02}.npz'
        receipt=f'{split}-{part:02}.json'
        if (OUT/receipt).is_file():
            saved=R.read(receipt)
            assert saved['sha']==C.A.digest(OUT/name) and saved['plan']==C.A.digest(OUT/'plan.json')
            results.append(name);continue
        check_write(OUT/name,70*2**20,operation='四范式逐批空间复查')
        panels=[]
        for b,packet in enumerate(batched(group,BATCH)):
            panel=C.packet(model,packet)
            panel['q']=t.tensor([row['query'] for row in packet],device='cuda')
            panel['keep_field']=split=='confirm' and part==0 and b<4
            panel['intervene']=split=='confirm' and part<4
            panels.append(panel)
        total=40*math.ceil(len(rows)/SLAB)
        C.N.progress(split,part*40,total,started)
        for index in range(40):
            with model.block(index) as block:
                for panel in panels:
                    x,mask=panel['x'].cuda(),panel['mask'].cuda()
                    angles=[p.cuda() for p in panel['angles']]
                    q,k,v=model.qkv(block,x,angles)
                    o=model.attention(block,q,k,v,mask)
                    query=panel['q']
                    if panel['intervene'] and index>=candidate['block']:
                        native_step(model,block,index,panel,x,k,v,mask,angles,query,candidate)
                    panel['x']=conditional_write(model,block,index,panel,x,q,k,v,o,mask,candidate,query).cpu()
            if (index+1)%5==0:
                C.N.progress(split,part*40+index+1,total,started)
        observed=[]; fusion=[];pool=[]
        for panel in panels:
            query=panel['q']
            j=t.arange(len(query),device='cuda')
            x=panel['x'].cuda()[j[:,None],query]
            logits=model.logits(x).float()
            top=logits.argmax(-1).cpu().numpy()
            raw=probe(model,panel,logits,setup_weights)
            row=dict(ids=np.array([r['id'] for r in panel['rows']]),top=top,margin=raw,gua=panel['gua'],mass=panel['mass'])
            pool.extend(base.bit_pattern(logits[:,0].to(t.bfloat16)))
            if len(pool)==16:
                z=np.stack(pool)
                zl=t.as_tensor(z,device='cuda').view(t.bfloat16).float().cpu().numpy()
                fusion.append(fusion_stats(model,zl))
                pool=[]
            if panel['keep_field']:
                row['field']=np.stack((*panel['field'],base.bit_pattern(x[:,[0,3]])),1)
                row['field_ids']=row['ids']
            if panel['intervene']:
                edited=model.logits(panel['live'].cuda()).float()
                assert t.equal(edited[0],edited[5])
                row['edit_margin']=opening_scores(model,edited,setup_weights[:2]).T.cpu().numpy()
                row['edit_top']=edited.argmax(-1).T.cpu().numpy()
                row['head_doses']=panel['head_doses']
                row['mlp_doses']=panel['mlp_doses']
                row['native_gua']=panel['native_gua']
            observed.append(row)
        assert not pool
        arrays={key:np.concatenate([p[key] for p in observed if key in p]) for key in set().union(*observed)}
        arrays['fusion']=np.stack(fusion)
        sha=base.write_np(name,arrays)
        R.write(receipt,dict(file=name,sha=sha,plan=C.A.digest(OUT/'plan.json'),views=len(group),at=C.stamp()))
        results.append(name)
    return results


def run():
    import torch
    from paradigm_four_report import analyze

    plan=verify()
    setup()
    candidate=R.read('candidate.json')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('four-load',0,2,start)
        assert C.Prior.identity()==plan['snapshot']
        model=C.N.Engine()
        C.N.progress('four-load',2,2,start)
        try:
            for split in ('pilot','confirm'):
                rows=list(M.expanded(model.tok,split))
                assert M.digest(rows)==plan['material'][split]['sha']
                result=measure(model,split,rows,plan,candidate)
                analyze(rows,result,plan,model.tok,split)
        finally:
            model.close()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','all'))
    args=cli.parse_args()
    R.OUT=OUT
    (freeze if args.action=='freeze' else run)()


if __name__=='__main__':
    try:main()
    except StorageBlocked as error:raise SystemExit(str(error)) from None
