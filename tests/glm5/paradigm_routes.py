"""Stream native 14B attention and MLP to connect language paradigms to actual writes."""

from itertools import batched
import argparse
import time

import numpy as np

import paradigm_routes_material as M
import record_update_run as R
from deploy.storage_policy import checked_dir,check_write,staging_file,StorageBlocked
from tools.power_analysis import check_phase


C = R.C
OUT = R.ROOT/'tests/glm5/result/paradigm_routes_20261117_v1'
TEMP = R.ROOT/'tests/glm5_temp/paradigm_routes_20261117_v1'
WIDTH = 160
BATCH = 8
SLAB = 256
PEAK = 6*2**30
REGIONS = ('request','reference','language','facts','guide','other')
BASE_SOURCES = R.source_hashes
FILES = ('paradigm_routes_material.py','paradigm_routes.py','paradigm_routes_report.py')
EARLY_LAYER = 20


def configure():
    R.OUT,R.TEMP = OUT,TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH = M.RUN,M.SEED,WIDTH
    R.BATCH,R.SLAB = BATCH,SLAB
    R.configure(PEAK)


def sources():
    all_ = set(BASE_SOURCES())|{f'tests/glm5/{file}' for file in FILES}
    return {file:C.A.digest(R.ROOT/file) for file in sorted(all_)}


def write_np(name,arrays):
    budget=sum(np.asarray(value).nbytes for value in arrays.values())+2**18
    target=check_write(OUT/name,budget,operation='原生范式路径数据')
    with staging_file(target) as temp,temp.open('wb') as file:
        np.savez_compressed(file,**arrays)
    return C.A.digest(target)


def read_np(name):
    with np.load(OUT/name,allow_pickle=False) as file:
        return {key:file[key] for key in file.files}


def freeze():
    from transformers import AutoTokenizer

    if (OUT/'plan.json').exists():
        raise FileExistsError('Existing frozen paradigm study')
    checked_dir(OUT,PEAK,operation='范式和原生神经元联合研究')
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    material=M.inspect(tok)
    p=dict(run=M.RUN,at=C.stamp(),provenance=R.metadata(),snapshot=C.Prior.identity(),versions=R.versions(),
           sources=sources(),material=material,anchors=M.alphabet(tok),budget=dict(bytes=PEAK,reserve_gib=20,volume='D:'),
           power=check_phase(.20,M.SIZES['confirm'],.20,.01),
           dimensions=dict(layers=40,heads=40,mlp_units=17408,hidden_width=5120),
           design=dict(question='Link native neurons to attention selection and fusion/competition of requested format, contextual reference and language while content/count and factual selector remain shared.',
                       matrix='2 requested formats academic/Python × 2 archive formats academic/Python × 2 answer languages English/Chinese. Same prompt length and all cue/span/query positions in each family. Hold out content numbers and frames; joint lockbox never sampled.',
                       measurements='All 40 heads at all 40 layers at actual final query. Source-group attention, unnormalized source-group logsumexp scores, head write projected on native output print versus The. Original G/U/A and signed Wdown·(Wprint-WThe) for every MLP unit, plus embedding and every hidden boundary at query. Full last-vocabulary logits and a separate shared print quote prefix readout. Native BF16 sequential layers and exact RMS/QK normalizers/RoPE.',
                       behavior='Read native first all-vocabulary argmax at free answer boundary. Require Python print hit >=.75 each reference/language and first-token Chinese-vs-English odds aligned in >=.75 family conditions. This only calibrates format onset. Supplied print quote conditional query separately measures language, and requires correct language-script top1 >=.75 per target/reference/language. Full answer correctness is unmeasured.',
                       fit='After discovery: freeze 32 MLP units with largest family-mean absolute exact Wdown readout contribution difference between requested academic and Python, averaged across the other factors, layers eligible 12..31. Select single attention head at 8..31 with largest native projected write gap on same contrast. Same-layer count-matched random units and different-GQA head control fixed before confirmation.',
                       selection='Use signed exact local projections only to select candidates. Text baseline uses all visible request/reference/language, positions, factual selector and quantity. Native model adds 32 selected A and selected head summary to forecast final normalized state, same rows and ridge=12. Dictionary comparator deferred as additional cost. Discovery-only freeze then unopened confirm predictions and intervention.',
                       forecast='Ridge forecast final normalized hidden and then original unembedding. Require >=10% family-mean full-vocabulary KL reduction vs explicit text baseline, family 99% bootstrap lower>0, at least 64 heldout families. Use 32 selected original units or 32 matched random units. Selection confined to layers <=31. Natural response forecasting remains descriptive if pilot fails.',
                       interpretation='Exact projections are local calculations and contextual terms, uncertainty remains in downstream normalization and parallel routes. A language-specific head label requires unseen-frame predictive or intervention evidence; source-group labels are experimental regions.',
                       intervention='If onset/conditional gates fail, collect and analyze internal natural response structure for all discovery/confirm as exploratory only; no advancing failed behavior as a positive functional circuit. If both behavior and forecast gates pass, at selected query replay same-layer native head knockout and matched dose control, native self-restore, then downstream recompute. Separate component effect from actual top1 format switches.',
                       runtime='Original BF16 14B block-streamed to CUDA. Write <=5 minute progress and shared VRAM via DXGI. Read only fixed shared final query at two positions, causal inputs only. No generation or label clamping used for onset query.'))
    print('Frozen',R.write('plan.json',p),p['snapshot']['snapshot'],flush=True)


def verify():
    p=R.read('plan.json')
    assert p['sources']==sources() and p['versions']==R.versions()
    return p


def variants(tok,split,conditional=False):
    data=list(M.material(tok,split))
    for row in data:
        if conditional:
            row['token']=dict(ids=row['prefix_code'],slots=[len(row['prefix_code'])-1])
    return data


def response_groups(model,p):
    t=model.t
    if 'locations' in p:
        return p['locations']
    groups=t.zeros((len(p['rows']),len(REGIONS),WIDTH),device='cuda',dtype=t.bool)
    for j,row in enumerate(p['rows']):
        for i,region in enumerate(REGIONS[:-1]):
            groups[j,i,row['spans'][region]]=True
        n=len(row['token']['ids'])
        groups[j,-1,:n]=~groups[j,:-1,:n].any(0)
    p['locations']=groups
    return groups


def bit_pattern(tensor):
    import torch

    return tensor.contiguous().view(torch.int16).cpu().numpy().copy()


def trace(model,block,index,p,x,q,k,v,o,mask):
    t=model.t
    j=t.arange(len(x),device='cuda')
    c=p['c'].cuda()
    current=x[j,c]
    p.setdefault('field',[]).append(bit_pattern(current))
    if index==EARLY_LAYER:
        p['early']=current.float().cpu().numpy()
    visible=mask[j,:,c].unsqueeze(2)
    query=q[j,:,c].unsqueeze(2)
    distribution=model.read(query,k,visible)
    locations=response_groups(model,p)
    masses=(distribution[:,:,None,:]*locations[:,None]).sum(-1)
    keys=k.repeat_interleave(model.groups,dim=1)
    score=(query.float()*keys.float()).sum(-1)*128**-.5
    source_score=t.stack([score.masked_fill(~locations[:,i,None,:],-t.inf).logsumexp(-1) for i in range(len(REGIONS))],-1)
    p.setdefault('alpha',[]).append(masses.cpu().numpy())
    p.setdefault('scores',[]).append(source_score.cpu().numpy())
    weights=block.self_attn.o_proj.weight.reshape(model.width,model.heads,128)
    output_direction=p['output_direction']
    head_direction=t.einsum('d,dhk->hk',output_direction,weights.float())
    head_effect=(o[j,c].float()*head_direction[None]).sum(-1)
    p.setdefault('head_effect',[]).append(head_effect.cpu().numpy())
    projected=model.projections(block,tuple(range(model.heads)),o[j,c])
    p.setdefault('head_norm',[]).append(projected.norm(dim=-1).cpu().numpy())
    attn_out=block.self_attn.o_proj(o.reshape(*x.shape[:-1],model.width).contiguous())
    after=x+attn_out
    normalized=block.post_attention_layernorm(after)
    gate=block.mlp.gate_proj(normalized)
    up=block.mlp.up_proj(normalized)
    activation=t.nn.functional.silu(gate)*up
    native_direction=output_direction@block.mlp.down_proj.weight.float()
    p.setdefault('G',[]).append(bit_pattern(gate[j,c]))
    p.setdefault('U',[]).append(bit_pattern(up[j,c]))
    p.setdefault('A',[]).append(bit_pattern(activation[j,c]))
    p.setdefault('down_direction',[]).append(native_direction.cpu().numpy())
    result=after+block.mlp.down_proj(activation)
    if p['rows'][0]['split']=='pilot' and index in (0,20,39):
        native=model.residual(block,x,o)
        assert t.equal(result,native),(index,(result-native).abs().max().item())
    return result


def output(model,p):
    t=model.t
    x=p['x'].cuda()[t.arange(len(p['rows']),device='cuda'),p['c'].cuda()]
    norm=model.decoder.norm(x)
    logits=model.model.lm_head(norm)
    lp=logits.float().log_softmax(-1)
    tokens=logits.argmax(-1).cpu().numpy()
    compact={key:np.stack(p[key],axis=1) for key in ('alpha','scores','head_effect','head_norm','G','U','A')}
    field=np.stack((*p['field'],bit_pattern(x)),axis=1)
    down=np.stack(p['down_direction'])
    n=logits.shape[-1]
    script=np.asarray([any('\u3400'<=char<='\u9fff' for char in token) for token in model.tok.batch_decode([[i] for i in range(n)])]) if 'script_ids' not in model.__dict__ else model.script_ids
    model.script_ids=script
    mask=t.as_tensor(script,device='cuda')
    category=t.stack((lp[:,mask].logsumexp(-1),lp[:,~mask].logsumexp(-1)),-1)
    return dict(ids=np.array([r['id'] for r in p['rows']]),**compact,field=field,early=p['early'],
                target=bit_pattern(norm),logits=bit_pattern(logits),top=tokens,script=category.cpu().numpy(),
                down=down)


def take(model,split,conditional=False):
    rows=variants(model.tok,split,conditional)
    label=split+('-conditional' if conditional else '')
    start=time.monotonic()
    anchors=M.alphabet(model.tok)
    direction=(model.model.lm_head.weight[anchors['print']]-model.model.lm_head.weight[anchors['The']]).float()
    files=[]
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'{label}-{part:03}.npz'
        receipt=f'{label}-{part:03}.json'
        if (OUT/receipt).exists():
            old=R.read(receipt)
            assert old['sha']==C.A.digest(OUT/name)
            files.append(name)
            continue
        panels=[]
        for batch in batched(group,BATCH):
            packed=C.packet(model,batch)
            packed['output_direction']=direction
            panels.append(packed)
        for i in range(40):
            with model.block(i) as layer:
                for p in panels:
                    x,mask=p['x'].cuda(),p['mask'].cuda()
                    q,k,v=model.qkv(layer,x,[y.cuda() for y in p['angles']])
                    o=model.attention(layer,q,k,v,mask)
                    if conditional:
                        if i==EARLY_LAYER:
                            p['early']=x[model.t.arange(len(x),device='cuda'),p['c'].cuda()].float().cpu().numpy()
                        p['x']=model.residual(layer,x,o).cpu()
                    else:
                        p['x']=trace(model,layer,i,p,x,q,k,v,o,mask).cpu()
            if (i+1)%5==0:
                C.N.progress(label,part*40+i+1,40*((len(rows)+SLAB-1)//SLAB),start)
        if conditional:
            pieces=[]
            for p in panels:
                x=p['x'].cuda()[model.t.arange(len(p['rows']),device='cuda'),p['c'].cuda()]
                y=model.logits(x)
                pieces.append(dict(ids=np.array([r['id'] for r in p['rows']]),top=y.argmax(-1).cpu().numpy(),logits=bit_pattern(y)))
        else:
            pieces=[output(model,p) for p in panels]
        arrays={key:np.concatenate([piece[key] for piece in pieces]) for key in pieces[0] if key!='down'}
        if not conditional:
            assert all(np.array_equal(pieces[0]['down'],piece['down']) for piece in pieces)
            arrays['down']=pieces[0]['down']
        sha=write_np(name,arrays)
        R.write(receipt,dict(at=C.stamp(),file=name,sha=sha,plan=C.A.digest(OUT/'plan.json'),rows=len(group),conditional=conditional))
        files.append(name)
        C.N.progress(label+'-saved',min(len(rows),(part+1)*SLAB),len(rows),start)
    return dict(rows=rows,files=files)


def run():
    import torch

    p=verify()
    configure()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        started=time.monotonic()
        C.N.progress('paradigm-load',0,2,started)
        assert C.Prior.identity()==p['snapshot']
        C.N.progress('paradigm-load',1,2,started)
        model=C.N.Engine()
        C.N.progress('paradigm-load',2,2,started)
        try:
            from paradigm_routes_report import pilot,freeze_candidates,confirm

            a=take(model,'pilot')
            b=take(model,'pilot',True)
            gate=pilot(p,a,b,model.tok)
            take(model,'discovery')
            frozen=freeze_candidates(p,model.tok)
            take(model,'confirm')
            confirm(p,frozen,model.tok,model)
            R.write('execution.json',dict(at=C.stamp(),plan=C.A.digest(OUT/'plan.json'),pilot=gate,
                     stage='natural_response_maps_completed',candidate_freeze=C.A.digest(OUT/'candidate-freeze.json')))
        finally:
            model.close()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','all','close'))
    args=cli.parse_args()
    R.OUT,R.TEMP=OUT,TEMP
    if args.action=='freeze':
        freeze()
    elif args.action=='all':
        run()
    else:
        from paradigm_routes_report import close

        close(verify())


if __name__=='__main__':
    try:
        main()
    except StorageBlocked as error:
        raise SystemExit(str(error)) from None
