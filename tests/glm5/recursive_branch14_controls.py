"""Prespecified control comparisons, reported descriptively without new gates."""
from collections import defaultdict
import numpy as np
import recursive_branch14 as R
from recursive_branch14_analysis import inference


def aggregate(rows,key):
    families=defaultdict(list)
    for r in rows:families[r['family']].append(r[key])
    return inference([np.mean(v) for v in families.values()])


def joint_state():
    from scipy.special import expit
    allrows=R.materials();summary=[]
    for split in R.SPLITS[1:]:
        meta=R.read(R.OUT/split/'collection.json');rows=[allrows[i] for i in meta['row_indices']]
        for h,(b,u) in R.CANDIDATES.items():
            values=np.column_stack([R.unbits(np.load(R.OUT/split/f'B{b}-{field}.npy',mmap_mode='r')[:,0,u]) for field in 'GVA']).astype(np.float64)
            for role in ['operative','quoted','inventory']:
                indices=[i for i,r in enumerate(rows) if r['hypothesis']==h and r['role']==role]
                g,v,a=values[indices].T;s=g*expit(g)
                covariance=float(np.mean((s-s.mean())*(v-v.mean())))
                product=float(s.mean()*v.mean());exact=float(np.mean(s*v))
                assert abs(product+covariance-exact)<1e-10
                quadrants={}
                for sg,sv,name in [(1,1,'++'),(1,-1,'+-'),(-1,1,'-+'),(-1,-1,'--')]:
                    mask=((g>=0) if sg>0 else (g<0))&((v>=0) if sv>0 else (v<0))
                    quadrants[name]=dict(rows=int(mask.sum()),native_abs_A_share=float(np.abs(a[mask]).sum()/max(np.abs(a).sum(),1e-30)))
                summary.append(dict(hypothesis=h,split=split,role=role,rows=len(indices),
                    mean_G=float(g.mean()),mean_V=float(v.mean()),mean_native_A=float(a.mean()),
                    product_of_mean_siluG_and_meanV=product,conditional_covariance=covariance,
                    fp64_mean_siluG_timesV=exact,native_mean_minus_fp64=float(a.mean()-exact),
                    silu_meanG_times_meanV=float(g.mean()*expit(g.mean())*v.mean()),quadrants=quadrants))
    R.save('joint-state-summary.json',dict(kind='real',summary=summary,
        source_sha256=R.file_sha(__file__),status='Post-primary descriptive arithmetic; no feature, cutoff, readout, or intervention changed',
        input_collections={split:R.file_sha(R.OUT/split/'collection.json') for split in R.SPLITS[1:]},
        formula='mean(SiLU(G)*V)=mean(SiLU(G))*mean(V)+cov_population(SiLU(G),V)',
        precision='G,V,A read from native BF16 arrays. Identity uses FP64 SiLU of stored G; native SiLU/product rounding is explicitly retained as a residual.',
        inference='No new hypothesis test; role/split means pool paired family views and do not claim independent rows.'))


def run():
    R.configure();R.admission();results=[]
    for h in R.CANDIDATES:
        data=R.read(R.OUT/f'intervention-{h}.json')
        rows=data['rows'];lookup={(r['id'],r['arm']):r for r in rows}
        target=[r for r in rows if r['hypothesis']==h]
        group=[dict(r,contrast=r['delta_margin']-lookup[r['id'],'matched_group_0']['delta_margin'])
               for r in target if r['arm']=='group_0']
        results.append(dict(hypothesis=h,contrast='group0 minus matched-group0',scope='target',
                            comparison=aggregate(group,'contrast'),interpretation='Negative means selected-group removal damages margin more than matched-group removal; does not alone identify a semantic operation.'))
        for arm in ['single_0','group_0','matched_group_0']:
            by=defaultdict(dict)
            for r in target:
                if r['arm']==arm:by[r['family']][r['role'],r['lexical']]=r['delta_margin']
            delta=[]
            for f,v in by.items():
                assert len(v)==6
                delta.append(sum((v['operative',l]+v['quoted',l])/2-v['inventory',l] for l in [0,1])/2)
            results.append(dict(hypothesis=h,arm=arm,scope='target',contrast='relation minus inventory intervention effect',
                comparison=inference(delta),interpretation='Within-family differential margin change; question difficulty and wording remain potential explanations.'))
    R.save('control-comparisons.json',dict(kind='real',results=results,source_sha256=R.file_sha(__file__),
        inference='95% exploratory family intervals for control contrasts anticipated in the plan. No additional pass threshold or mechanism promotion.',
        inputs={h:R.file_sha(R.OUT/f'intervention-{h}.json') for h in R.CANDIDATES}))
    joint_state()


if __name__=='__main__':run()
