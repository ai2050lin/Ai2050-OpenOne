"""Frozen family-level paired tests and full-vocabulary descriptive diagnostics."""
import collections,json,time
import numpy as np
import b36_switch14_design as D
M=D.M;S=D.S


def decode(a):return (a.astype(np.uint32)<<16).view(np.float32)


def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]);out=[None]*len(ps);last=0
    for j,i in enumerate(order):last=max(last,min(1,(len(ps)-j)*ps[i]));out[i]=last
    return out


def affine(ops):
    a,b=1,0
    for op,k in ops:
        if op=='add':b+=k
        elif op=='sub':b-=k
        else:a*=k;b*=k
    return a,b


def run():
    from scipy.special import logsumexp
    from scipy.stats import binomtest
    S.configure();p=S.read(S.OUT/'plan.json');M.admit(p);nat=S.read(S.OUT/'native-summary.json')
    assert nat['plan_sha256']==S.file_sha(S.OUT/'plan.json')
    assert nat['behavior_analysis_sha256']==S.file_sha(S.OUT/'behavior-analysis.json')
    records=[];field_views={};start=time.monotonic();arms=p['native_conditions'];cl=p['classes'];class_by_token={i:k for k,ids in cl.items() for i in ids}
    for batch,item in enumerate(nat['records']):
        assert item['sha256']==S.file_sha(S.OUT/item['path'])
        with np.load(S.OUT/item['path']) as z:
            logits=z['logits_baseline_and_xor'].copy();logits[1:]^=logits[0]
            vals=decode(logits).astype(np.float64);logp=vals-logsumexp(vals,axis=-1,keepdims=True)
            assert np.array_equal(logits[0],logits[1])==item['checks']['noop_full_batch_logits_bitwise'] or not item['checks']['noop_full_batch_logits_bitwise']
            after36=decode(z['after36']).astype(np.float64);h=z['hidden'].copy()
            for j,event in enumerate(item['events']):
                baseline=vals[0,j];sort=np.partition(baseline,-2);target=cl[event['target_class']]
                row=dict(event,conditions={},baseline_top2_gap=float(sort[-1]-sort[-2]),baseline_entropy=float(-(np.exp(logp[0,j])*logp[0,j]).sum()))
                for ai,arm in enumerate(arms):
                    top=int(vals[ai,j].argmax());prob=np.exp(logp[ai,j])
                    row['conditions'][arm]=dict(top1=top,top1_class=class_by_token.get(top,'other'),correct=class_by_token.get(top,'other')==event['target_class'],
                        target_log_probability=float(logsumexp(logp[ai,j,target])),
                        class_probabilities={c:float(np.exp(logsumexp(logp[ai,j,ids]))) for c,ids in cl.items()},
                        kl_from_baseline=float((np.exp(logp[0,j])*(logp[0,j]-logp[ai,j])).sum()),
                        actual_after36_displacement_norm=float(np.linalg.norm(after36[ai,j]-after36[0,j])))
                assert row['conditions']['baseline']['top1']==event['natural_token']
                records.append(row);field_views[(event['family'],event['arm'],event['site'])]=h[j]
        S.log('native_analysis',batch+1,len(nat['records']),start)
    expected={(r['id'],site) for r in p['rows'] for site in D.SITES}
    assert {(r['id'],r['site']) for r in records}==expected
    families=[g['id'] for g in p['families']];by=collections.defaultdict(list)
    for r in records:by[r['family']].append(r)
    joint={arm:np.asarray([all(r['conditions'][arm]['correct'] for r in by[f]) for f in families]) for arm in arms}
    rng=np.random.default_rng(S.SEED);boot=rng.integers(0,96,size=(10000,96));prim=[]
    groups=collections.defaultdict(list)
    for i,g in enumerate(p['families']):groups[(affine(g['ops'][:2]),affine(g['ops']))].append(i)
    group_indices=list(groups.values());cluster_draw=rng.integers(0,len(groups),size=(4000,len(groups)))
    for left,right in p['primary']:
        x=joint[left];y=joint[right];gain=int((~x&y).sum());loss=int((x&~y).sum());diff=y.astype(float)-x
        ci=np.quantile(diff[boot].mean(1),[.025,.975]).tolist()
        cluster=np.asarray([diff[np.concatenate([group_indices[k] for k in ids])].mean() for ids in cluster_draw])
        prim.append(dict(left=left,right=right,n=96,gained=gain,lost=loss,delta=float(diff.mean()),admitted=nat['passed'],
            p_exact=float(binomtest(min(gain,loss),gain+loss,.5).pvalue) if gain+loss and nat['passed'] else 1.,family95=ci,
            affine_pair_clusters=len(groups),affine_pair_cluster95=np.quantile(cluster,[.025,.975]).tolist()))
    for r,pv in zip(prim,holm([r['p_exact'] for r in prim])):r['p_holm']=pv
    cells=[]
    for interface in D.ARMS:
        for site in D.SITES:
            rr=[r for r in records if r['arm']==interface and r['site']==site]
            for arm in arms:
                cells.append(dict(interface=interface,site=site,condition=arm,n=len(rr),correct=sum(r['conditions'][arm]['correct'] for r in rr),
                    changed_token=sum(r['conditions'][arm]['top1']!=r['conditions']['baseline']['top1'] for r in rr),
                    target_logp_delta_mean=float(np.mean([r['conditions'][arm]['target_log_probability']-r['conditions']['baseline']['target_log_probability'] for r in rr])),
                    kl_mean=float(np.mean([r['conditions'][arm]['kl_from_baseline'] for r in rr])),
                    displacement_norm_mean=float(np.mean([r['conditions'][arm]['actual_after36_displacement_norm'] for r in rr]))))
    heat={};embedding_equal={}
    for site in D.SITES:
        a=np.stack([decode(field_views[(f,'first2',site)]) for f in families]);b=np.stack([decode(field_views[(f,'first3',site)]) for f in families])
        heat[site+'_first2']=a.mean(0);heat[site+'_first3']=b.mean(0);heat[site+'_paired_difference']=(b-a).mean(0)
        heat[site+'_rms']=np.sqrt((a.astype(np.float64)**2+b.astype(np.float64)**2).mean(axis=(0,2))/2)
        embedding_equal[site]=int(np.all(a[:,0]==b[:,0],axis=1).sum())
    np.savez_compressed(S.OUT/'native-heatmaps.npz',**heat)
    S.save('native-analysis.json',dict(admitted=nat['passed'],n_families=96,n_events=len(records),primary=prim,cells=cells,
        family_joint_correct={k:int(v.sum()) for k,v in joint.items()},events=records,embedding_pair_bitwise_equal=embedding_equal,
        baseline_top2_gap=dict(min=min(r['baseline_top2_gap'] for r in records),median=float(np.median([r['baseline_top2_gap'] for r in records])),max=max(r['baseline_top2_gap'] for r in records)),
        source_sha256=S.file_sha(__file__),native_summary_sha256=S.file_sha(S.OUT/'native-summary.json'),
        caveats='No-op/behavior/rerun admission is separate from effects. Joint class correctness spans6 events per family; correlated views never increaseN. These are next-token classes, not full continuations or unique necessary mechanisms. Baseline next token defines the observational event and is excluded from its causal input. Whole-class probabilities may include token strings containing additional characters. Float64 KL from stored BF16 logits. Zero-discordance bootstrap degeneracy does not prove equivalence. Affine-pair clustering is sensitivity only.'))
    print('Native analysis',nat['passed'],{k:int(v.sum()) for k,v in joint.items()},flush=True)


if __name__=='__main__':run()
