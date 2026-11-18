"""Descriptive holdout profiles for preselected groups; no change to primary gates."""
import numpy as np
from scipy.stats import nct
from scipy.optimize import brentq
import branch_reuse14 as B
import branch_reuse14_analysis as A
import branch_reuse14_qualified as Q


def dz_interval(values,alpha):
    values=np.asarray(values);n=len(values);sd=values.std(ddof=1)
    if sd==0:return None
    t=values.mean()/sd*np.sqrt(n);bound=max(200.,abs(t)+100.)
    lo=brentq(lambda nc:nct.cdf(t,n-1,nc)-(1-alpha/2),-bound,bound)
    hi=brentq(lambda nc:nct.cdf(t,n-1,nc)-alpha/2,-bound,bound)
    return [float(lo/np.sqrt(n)),float(hi/np.sqrt(n))]


def run():
    Q.bind();selection=B.read(B.OUT/'frozen-selection.json');rows,_=A.load('confirm');cards=[];group_report=[];bounds=[]
    groups=[('B19-preselected',19,selection['group']['units']),('B39-roots',39,selection['graphs']['correlation']['roots']),('B26-historical',26,[13099])]
    for name,block,units in groups:
      values=A.target('confirm',block,units).reshape(len(rows),len(units),3)
      for index,unit in enumerate(units):
        records=[]
        for branch in A.BRANCHES:
          ix=[i for i,r in enumerate(rows) if r['branch']==branch];rr=[rows[i] for i in ix]
          for f,field in enumerate('GVA'):
            delta=A.paired(rr,values[ix,index,f]);stat=A.inference(delta,.05/(len(units)*len(A.BRANCHES)))
            strata=[float(np.mean(values[[i for i in ix if rows[i]['role']=='rule' and rows[i]['title']==t],index,f])-np.mean(values[[i for i in ix if rows[i]['role']=='record' and rows[i]['title']==t],index,f])) for t in [0,1]]
            records.append(dict(branch=branch,field=field,contrast=stat,title_differences=strata))
            if block==19 and unit==64 and field in 'VA':bounds.append(dict(branch=branch,field=field,dz=stat['dz'],dz_interval=dz_interval(delta,.05/6),
                lower_tail_probability_at_dz_point5=float(nct.cdf(stat['dz']*len(delta)**.5,len(delta)-1,.5*len(delta)**.5)),
                status='Supplemental effect-size bound using the already frozen minimum dz=.5; not a new primary pass test.'))
        arows=[r for r in records if r['field']=='A']
        group_report.append(dict(group=name,block=block,unit=unit,A_results=arows,
            positive_both_eligible_branches=all(r['contrast']['mean']>0 and r['contrast']['ci'][0]>0 and min(r['title_differences'])>0 for r in arows)))
        rank=np.argsort(values[:,index,2]);examples=[]
        for i in [int(rank[0]),int(rank[-1])]:examples.append(dict(id=rows[i]['id'],body=rows[i]['body'],A=float(values[i,index,2]),
            source='Constructed confirmation material, selected descriptively after evaluation; not natural-corpus extremes.'))
        cards.append(dict(feature_id=f'{B.SNAPSHOT}:B{block}:mlp_down_input:U{unit}:BR14v3',model_snapshot=B.SNAPSHOT,block=block,unit=unit,
            readout='content end before question/options',method='preselected native unit; group '+name,
            natural_activation_statistics='Not measured on new natural corpus in this trial. Historical candidates retain prior feature cards; no new natural corpus claim.',
            template_selectivity=records,examples=examples,unembedding_readout='Not individually recomputed; graph output uses a fitted group reader then native final norm/unembedding, separately evaluated.',
            causal_test='B19 group has original-model group intervention; only U64 has individual intervention. B39 roots measured downstream of B36 edits. Group effects cannot be assigned to each member.',
            upstream_downstream='Frozen graph indices, fits and intervention targets in frozen-selection.json and intervention-B36/result.json; edges have separate observation and causal-effect grades.',
            counterexamples='All exceptions, failed primary tests and missing coordinate confirmation branch retained in linked reports.',
            evidence_grade='Candidate observation, preselected-unit holdout description; no unique semantic label or new formally confirmed global atlas node.'))
    B.save('group-response-profiles.json',dict(results=group_report,
        method='Fixed discovery-selected indices, independent confirmation families, content-end A paired by family and averaged across title. Intervals correct within each fixed group across its two eligible branches; G/V descriptive. No altered primary criterion and no new independent samples.',
        limitations='Shared positive responses can reflect common document-role wording, task format or conditional mode. Scalar A signs across units do not establish aligned physical writes or a higher-complexity feature.',source_sha256=B.file_sha(__file__)))
    B.save('feature-cards.json',dict(cards=cards,source_sha256=B.file_sha(__file__)))
    B.save('primary-effect-bounds.json',dict(results=bounds,source_sha256=B.file_sha(__file__),
        interpretation='A positive effect no smaller than the frozen floor can be contradicted even when a negative effect is not significant versus zero. These supplemental intervals do not repair missing branches, prediction failure or causality.'))


if __name__=='__main__':run()
