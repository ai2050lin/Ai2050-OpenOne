"""Explicit strict scoring, matched controls and post-confirmation counterexamples."""
import numpy as np
from scipy.special import expit
import branch_reuse14 as B
import branch_reuse14_analysis as A


def strict(row,key='scores'):
    s=row[key];g=row['gold'];return s[g]>max(v for j,v in enumerate(s) if j!=g)


def family_inference(rows,key):
    families=sorted({r['family'] for r in rows})
    return A.inference([np.mean([r[key] for r in rows if r['family']==f]) for f in families]) if families else None


def run():
    B.configure();comparisons=[];accuracies=[]
    for kind in ['B19','B36']:
      data=B.read(B.OUT/f'intervention-{kind}/result.json');rows=data['rows'];by={(r['arm'],r['id']):r for r in rows}
      pairs=([('group_0','matched_0'),('group_2','matched_2'),('single_0','single_norm'),('group_0','group_norm')] if kind=='B19' else
             [(f'{m}_{d}',f'{m}_matched_{d}') for m in A.METHODS for d in [0,2]])
      for branch in A.BRANCHES:
        for arm in data['arms']:
          rr=[r for r in rows if r['branch']==branch and r['arm']==arm]
          accuracies.append(dict(kind=kind,branch=branch,arm=arm,rows=len(rr),strict_accuracy=float(np.mean([strict(r) for r in rr])),
            strict_baseline_accuracy=float(np.mean([strict(r,'baseline_scores') for r in rr])),
            ties=sum(r['scores'][r['gold']]==max(v for j,v in enumerate(r['scores']) if j!=r['gold']) for r in rr)))
        for native,control in pairs:
          rr=[r for r in rows if r['branch']==branch and r['arm']==native];use=[]
          for r in rr:
            c=by[control,r['id']]
            if control.endswith('_norm') and not c['norm_control']['all_token_norm_matched']:continue
            use.append(dict(r,native_minus_control=r['delta_margin']-c['delta_margin']))
          comparisons.append(dict(kind=kind,branch=branch,native=native,control=control,qualified_rows=len(use),total_rows=len(rr),
            native_minus_control=family_inference(use,'native_minus_control'),
            interpretation='Matched-discovery-amplitude groups are not guaranteed equal realized interventions. Strict realized-norm subsets may be selected and are exploratory.'))
    rows,_=A.load('confirm');examples=[];joint=[]
    for name,block,unit in [('B19U64',19,64),('B26U13099',26,13099)]:
      y=A.target('confirm',block,[unit])
      for branch in A.BRANCHES:
        indices=[i for i,r in enumerate(rows) if r['branch']==branch];families=sorted({rows[i]['family'] for i in indices})
        delta=A.paired([rows[i] for i in indices],y[indices,2]);order=np.argsort(delta)
        for k in [int(order[0]),int(order[-1])]:
          family=families[k];ii=[i for i in indices if rows[i]['family']==family]
          examples.append(dict(candidate=name,branch=branch,family=family,selection='Lowest and highest observed paired A difference after formal evaluation; descriptive only.',A_difference=float(delta[k]),
            variants=[dict(id=rows[i]['id'],role=rows[i]['role'],title=rows[i]['title'],body=rows[i]['body'],
                future_probe=rows[i]['prompt'][len(rows[i]['body']):],G=float(y[i,0]),V=float(y[i,1]),A=float(y[i,2])) for i in ii]))
        for role in ['rule','record']:
          ix=[i for i in indices if rows[i]['role']==role];g,v,a=y[ix].T;s=g*expit(g)
          mean_product=float(s.mean()*v.mean());cov=float(np.mean((s-s.mean())*(v-v.mean())))
          joint.append(dict(candidate=name,branch=branch,role=role,n=len(ix),mean_G=float(g.mean()),mean_V=float(v.mean()),mean_A=float(a.mean()),
            product_of_means=mean_product,covariance=cov,native_rounding_residual=float(a.mean()-mean_product-cov),
            negative_GV_absA_fraction=float(np.abs(a[(g<0)&(v<0)]).sum()/max(np.abs(a).sum(),1e-14))))
    B.save('control-comparisons.json',dict(comparisons=comparisons,strict_accuracies=accuracies,source_sha256=B.file_sha(__file__),
        multiplicity='All comparisons in this file exploratory; not substitutes for failed preregistered primary gates.'))
    B.save('counterexamples.json',dict(examples=examples,source_sha256=B.file_sha(__file__)))
    B.save('joint-state-summary.json',dict(results=joint,formula='E[A]=E[SiLU(G)]E[V]+Cov(SiLU(G),V)+native BF16 rounding residual',
        interpretation='Expectation decomposition is a mathematical identity, not an inferred semantic law. No post-hoc switch from primary V/A to covariance as a passing gate.',source_sha256=B.file_sha(__file__)))


if __name__=='__main__':run()
