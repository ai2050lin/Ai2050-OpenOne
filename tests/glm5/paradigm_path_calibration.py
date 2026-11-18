"""Reproduce the six Phase 2970 mismatches and calibrate same-shape positive doses."""

from itertools import batched
import time

import numpy as np

import paradigm_four_run as F
import paradigm_four_report as V
import paradigm_path_core as P
from deploy.storage_policy import checked_dir


R,C=F.R,F.C
PARENT=F.OUT
OUT=R.ROOT/'tests/glm5/result/paradigm_path_20261117_v1'
TEMP=R.ROOT/'tests/glm5_temp'/OUT.name
PEAK=256*2**20
TECHNICAL_MIN=.98


def setup():
    R.OUT,R.TEMP=OUT,TEMP
    R.M.RUN,R.M.SEED,R.M.WIDTH='PARADIGM-PATH-20261117-v1',29722026,F.M.WIDTH
    R.BATCH,R.SLAB=8,128
    R.configure(PEAK)
    F.base.OUT=OUT


def old_data(tokenizer):
    previous=R.OUT
    R.OUT=V.OUT=PARENT
    try:
        archived=V.data_load([f'confirm-{i:02}.npz' for i in range(4)])
        candidate=R.read('candidate.json')
    finally:
        R.OUT=previous
        V.OUT=PARENT
    families={}
    selected=set()
    for i,row in enumerate(F.M.expanded(tokenizer,'confirm')):
        if i>=len(archived['ids']):break
        assert archived['ids'][i]==row['id']
        families.setdefault(row['family'],[]).append((i,row))
        if archived['edit_top'][i,0]!=archived['top'][i,0]:
            selected.add(row['family'])
    assert len(selected)==6
    for family in families:
        if family not in selected:
            selected.add(family)
            if len(selected)==8:break
    indexes,rows=zip(*(pair for family,entries in families.items() if family in selected for pair in entries))
    return archived,np.asarray(indexes),list(rows),candidate


def spot_diff(t,left,right):
    return (left.float()-right.float()).reshape(len(left),-1).abs().amax(-1).cpu().numpy()


def benchmark(model,rows,candidate):
    t=model.t
    start=time.monotonic()
    panels=[C.packet(model,batch) for batch in batched(rows,8)]
    for p in panels:
        p['q']=t.tensor([r['query'] for r in p['rows']],device='cuda')
        p['layers']=[]
    for layer in range(40):
        with model.block(layer) as block:
            for p in panels:
                x=p['x'].cuda()
                angles=[a.cuda() for a in p['angles']]
                mask=p['mask'].cuda()
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,mask)
                j=t.arange(len(x),device='cuda')
                pos=p['q'][:,0]
                output=model.residual(block,x,o)
                if layer>=30:
                    F.native_step(model,block,layer,p,x,k,v,mask,angles,p['q'],candidate)
                    same=x[j,pos] if layer==30 else p['same'].cuda()
                    p['same']=P.same_shape_next(model,block,x,same,p,angles).cpu()
                    p['layers'].append(spot_diff(t,p['live'][0].cuda(),output[j,pos]))
                    assert t.equal(p['same'].cuda(),output[j,pos]),layer
                if layer==30:
                    oldq,oldk,oldv=model.qkv(block,x[j,pos].expand(8,-1,-1).reshape(8*len(x),1,model.width),
                                             [a[j,pos,None].repeat(8,1,1) for a in angles])
                    query=q[j,:,pos].unsqueeze(2)
                    vis=mask[j,:,pos].unsqueeze(2)
                    sliced=model.attention(block,query,k,v,vis)[:,0]
                    fixed=P.extract(model,block,x,o,pos)
                    assert t.equal(output,fixed['output'])
                    p['first_differences']=np.stack((spot_diff(t,oldq[:len(x)],query),
                        spot_diff(t,oldk[:len(x)],k[j,:,pos].unsqueeze(2)),
                        spot_diff(t,oldv[:len(x)],v[j,:,pos].unsqueeze(2)),spot_diff(t,sliced,o[j,pos])),axis=1)
                    control=P.source_cut(model,block,p,x,q,k,v,o,P.CONTROL,'archive')
                    doses=[];states=[]
                    for head in P.HEADS:
                        cut=P.source_cut(model,block,p,x,q,k,v,o,head,'archive')
                        selected,sd=P.neuron_restore(model,block,fixed,cut,candidate['selected'],'both')
                        random,rd=P.neuron_restore(model,block,fixed,cut,candidate['random'],'both')
                        *_,source=P.realized_match(fixed['after'][j,pos],cut['after'][j,pos],control['after'][j,pos],t,
                                                   (cut['direction'],control['direction']))
                        *_,restore=P.realized_match(cut['output'][j,pos],selected[j,pos],random[j,pos],t,(sd,rd))
                        doses.append(np.stack((source,restore),1))
                        states.append(F.base.bit_pattern(t.stack((
                            t.stack((fixed['after'][j,pos],cut['after'][j,pos],control['after'][j,pos]),1),
                            t.stack((cut['output'][j,pos],selected[j,pos],random[j,pos]),1),
                        ),1)))
                    p['doses']=np.stack(doses,1)
                    p['dose_endpoints']=np.stack(states,1)
                p['x']=output.cpu()
        if (layer+1)%5==0:
            C.N.progress('old-shape-dose-calibration',layer+1,40,start)
    results=[]
    for p in panels:
        x=p['x'].cuda()
        j=t.arange(len(x),device='cuda');pos=p['q'][:,0]
        fixed=P.full_logits(model,p,x)
        packed=model.logits(p['live'].cuda()).float()[0]
        same_h_wrong=model.logits(x[j,pos].expand(8,-1,-1).contiguous()).float()[0]
        changed=x.clone();changed[j,pos]=p['same'].cuda()
        repaired=P.full_logits(model,p,changed)
        assert t.equal(fixed,repaired)
        gap=fixed[:,0].topk(2,dim=-1).values
        results.append(dict(first_differences=p['first_differences'],slab=np.stack(p['layers'],1),
            full_top=fixed[:,0].argmax(-1).cpu().numpy(),old_top=packed.argmax(-1).cpu().numpy(),
            same_shape_top=repaired[:,0].argmax(-1).cpu().numpy(),onset_gap=(gap[:,0]-gap[:,1]).cpu().numpy(),
            unembedding_only_abs=spot_diff(t,same_h_wrong,fixed[:,0]),dose=p['doses'],
            dose_endpoints=p['dose_endpoints']))
    return {k:np.concatenate([r[k] for r in results]) for k in results[0]}


def run():
    import torch

    setup()
    if (OUT/'calibration.json').exists():
        raise FileExistsError('Calibration already persisted')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('calibration-load',0,2,start)
        snapshot=C.Prior.identity()
        model=C.N.Engine()
        C.N.progress('calibration-load',2,2,start)
        try:
            previous,indices,rows,candidate=old_data(model.tok)
            outputs=benchmark(model,rows,candidate)
            mismatch=previous['edit_top'][indices,0]!=previous['top'][indices,0]
            tokens=model.tok
            cases=[]
            for i in np.flatnonzero(mismatch):
                words=tokens.batch_decode([[int(outputs[k][i])] for k in ('full_top','old_top','same_shape_top')])
                cases.append(dict(id=rows[i]['id'],words=words,onset_native_gap=float(outputs['onset_gap'][i]),
                  qkv_and_sliced_attention=outputs['first_differences'][i].tolist(),
                  first_residual_difference_block=int(30+np.flatnonzero(outputs['slab'][i])[0]),
                  unembedding_only_abs=float(outputs['unembedding_only_abs'][i])))
        finally:
            model.close()
    assert np.array_equal(outputs['full_top'],previous['top'][indices,0])
    old_reproduced=outputs['old_top']==previous['edit_top'][indices,0]
    assert old_reproduced[mismatch].all(),old_reproduced[mismatch]
    equal=np.array_equal(outputs['full_top'],outputs['same_shape_top'])
    dose={}
    for axis,head in enumerate(P.HEADS):
        dose[str(head)]={name:P.dose_percentiles(outputs['dose'][:,axis,i])
                         for i,name in enumerate(('source','restore'))}
    prior=previous['mlp_doses']
    raw=np.abs(prior[:,0]-prior[:,1])/np.maximum(prior.max(1),.02)
    p95=float(np.quantile(raw,.95))
    sha=F.base.write_np('calibration.npz',outputs)
    outcome=dict(at=C.stamp(),snapshot=snapshot,source=C.A.digest(PARENT/'manifest.json'),artifacts=sha,
        ids=[r['id'] for r in rows],recorded_failures=cases,old_reproduced=int(old_reproduced.sum()),
        repaired_bit_equal=equal,previous_rescue_p95=p95,previous_native_means=prior.mean(0).tolist(),
        repaired_dose=dose,passed=bool(equal and min(d['coverage'] for a in dose.values() for d in a.values())>=TECHNICAL_MIN))
    R.write('calibration.json',outcome)
    print('Calibration',outcome['passed'],'old mismatches',len(cases),'dose',dose,flush=True)


if __name__=='__main__':
    run()
