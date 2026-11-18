"""Family-level effects, native paths and prospective local-operator scoring."""
import argparse,time
import numpy as np
from scipy.special import expit
import b36_composition14 as B
import b36_composition14_analysis as A
import b36_composition14_tail as T
from recursive_branch14_analysis import inference
from branch_reuse14_analysis import predict


def family_values(rows,values):
    return np.array([np.mean([values[i] for i,r in enumerate(rows) if r['family']==f]) for f in sorted({r['family'] for r in rows})])


def summarize(split='confirm'):
    B.configure();T.admit();rows,_=A.eligible(split);folder=B.OUT/f'tail-{split}';s=B.read(folder/'summary.json')
    assert s['row_ids']==[r['id'] for r in rows];start=time.monotonic()
    effect={(r['id'],r['arm']):r for r in s['effects']};arms=s['arms'];arm_ix={a:i for i,a in enumerate(arms)}
    summaries=[];paired=[];complexity=[];title_control=[];paths=[];prediction=[];unit_rows=[]
    for key in sorted({(r['branch'],r['depth'],r['mode']) for r in rows}):
        rr=[r for r in rows if (r['branch'],r['depth'],r['mode'])==key]
        for arm in arms:
            ee=[effect[r['id'],arm] for r in rr];family=family_values(rr,[e['delta_margin'] for e in ee]);stats=inference(family)
            acc=lambda e: e['scores'][e['gold']]>max(v for k,v in enumerate(e['scores']) if k!=e['gold'])
            summaries.append(dict(zip(['branch','depth','mode'],key),arm=arm,margin=stats,KL=float(np.mean([e['KL'] for e in ee])),strict_accuracy=float(np.mean([acc(e) for e in ee]))))
        values=[effect[r['id'],'zero']['delta_margin']-effect[r['id'],'matched_zero']['delta_margin'] for r in rr]
        stats=inference(family_values(rr,values),.05/12)
        paired.append(dict(zip(['branch','depth','mode'],key),contrast='zero minus matched_zero',stats=stats,passed=stats['ci'][1]<0))
        for arm in ['zero_hold_attention','zero_hold_mlp','zero_hold_both','zero_restore_roots','zero_restore_random_roots']:
            values=[effect[r['id'],arm]['delta_margin']-effect[r['id'],'zero']['delta_margin'] for r in rr]
            paths.append(dict(zip(['branch','depth','mode'],key),arm=arm,margin_recovery=inference(family_values(rr,values))))
        by={}
        for r in rr:by.setdefault(r['family'],{})[r['title']]=effect[r['id'],'zero']['delta_margin']
        values=[p[1]-p[0] for p in by.values() if set(p)=={0,1}]
        if values:title_control.append(dict(zip(['branch','depth','mode'],key),technical_minus_plain=inference(values)))
    for branch in sorted({r['branch'] for r in rows}):
        rr=[r for r in rows if r['branch']==branch];by={}
        for r in rr:by.setdefault(r['family'],{}).setdefault((r['depth'],r['mode']),[]).append(effect[r['id'],'zero']['delta_margin'])
        values=[]
        for fam,p in by.items():
            if all(k in p for k in [(2,'program'),(2,'archive'),(3,'program'),(3,'archive')]):
                values.append(np.mean(p[3,'program'])-np.mean(p[3,'archive'])-np.mean(p[2,'program'])+np.mean(p[2,'archive']))
        if values:
            stats=inference(values,.025);complexity.append(dict(branch=branch,stats=stats,passed=stats['ci'][1]<0 and stats['dz'] is not None and stats['dz']<=-.5))
    root_scale=A.get('roots-r1-selectedA')['ys'].reshape(len(B.roots()),3)
    target39=s['targets']['39'];index=[target39.index(u) for u in B.roots()]
    rowby={r['id']:r for r in rows};rowindex={r['id']:i for i,r in enumerate(rows)};metrics=[];nonlinear=[]
    learned_inputs=A.inputs(split,1);learned_models={k:A.get('roots-r1-'+k) for k in ['selectedA','hidden','hidden_prefix']}
    for chunk in sorted(folder.glob('chunk-*.json')):
        m=B.read(chunk);rr=[rowby[x] for x in m['row_ids']]
        assert B.file_sha(chunk.with_suffix('.npz'))==m['arrays_sha256']
        with np.load(chunk.with_suffix('.npz')) as d:
            raw=B.unbits(d['final_raw']).astype(np.float64);native=B.unbits(d['B39'][:,:,index,:]).astype(np.float64);jvp=d['root_jvp'];finalj=d['final_jvp']
            for dose,arm,col in [(0,'zero',0),(2,'double',4)]:
                actual=native[arm_ix[arm]]-native[0];standard=actual/root_scale
                loss_zero=np.mean(standard**2,axis=(1,2));loss_full=np.mean(((jvp[:,:,:,col]-actual)/root_scale)**2,axis=(1,2))
                loss_residual=np.mean(((jvp[:,:,:,col+3]-actual)/root_scale)**2,axis=(1,2))
                actualh=raw[arm_ix[arm]]-raw[0];fullh=finalj[:,:,(0 if dose==0 else 4)];seed=d['seed'][:,:,(0 if dose==0 else 1)]
                ix=np.array([rowindex[r['id']] for r in rr]);learned_losses={}
                for method,model in learned_models.items():
                    x=learned_inputs[method][ix];changed=x.copy()
                    if method=='selectedA':changed*=dose
                    else:changed[:,:B.DIM]+=seed
                    delta=(predict(model,changed)-predict(model,x)).reshape(-1,len(B.roots()),3)
                    learned_losses[method]=np.mean(((delta-actual)/root_scale)**2,axis=(1,2))
                for i,r in enumerate(rr):
                    metrics.append(dict(id=r['id'],family=r['family'],branch=r['branch'],depth=r['depth'],mode=r['mode'],title=r['title'],dose=dose,
                        root_zero=float(loss_zero[i]),root_full=float(loss_full[i]),root_residual=float(loss_residual[i]),
                        hidden_zero=float(np.mean(actualh[i]**2)),hidden_full=float(np.mean((fullh[i]-actualh[i])**2)),hidden_residual=float(np.mean((seed[i]-actualh[i])**2)),
                        **{'learned_'+k:float(v[i]) for k,v in learned_losses.items()}))
                # Finite gate/value/cross identity. BF16 A has a separate rounding remainder.
                g,v,a=np.moveaxis(native[0],-1,0);g1,v1,a1=np.moveaxis(native[arm_ix[arm]],-1,0)
                dg_silu=g1*expit(g1)-g*expit(g);dv=v1-v
                terms=[dg_silu*v,g*expit(g)*dv,dg_silu*dv]
                remainder=(a1-a)-sum(terms)
                for i,r in enumerate(rr):nonlinear.append(dict(id=r['id'],family=r['family'],branch=r['branch'],depth=r['depth'],mode=r['mode'],dose=dose,
                    actual_A_rms=float(np.sqrt(np.mean((a1[i]-a[i])**2))),gate_rms=float(np.sqrt(np.mean(terms[0][i]**2))),value_rms=float(np.sqrt(np.mean(terms[1][i]**2))),
                    cross_rms=float(np.sqrt(np.mean(terms[2][i]**2))),rounding_rms=float(np.sqrt(np.mean(remainder[i]**2)))))
        unit_rows+=m['unit_margin_linearization']
    for key in sorted({(r['branch'],r['depth'],r['mode'],r['dose']) for r in metrics}):
        rr=[r for r in metrics if (r['branch'],r['depth'],r['mode'],r['dose'])==key]
        ratios={k:A.grouped_ratio(rr,np.array([r['root_full'] for r in rr]),np.array([r['root_'+k] for r in rr])) for k in ['zero','residual']}
        rms=float(np.sqrt(np.mean([r['root_zero'] for r in rr])))
        prediction.append(dict(zip(['branch','depth','mode','dose'],key),rms_actual_standardized=rms,ratios=ratios,testable=rms>=.05,
            passed=rms>=.05 and ratios['zero']['ratio']<=.25 and ratios['residual']['ratio']<=.9,
            learned_baselines={k:dict(mse_over_zero=A.grouped_ratio(rr,np.array([r['learned_'+k] for r in rr]),np.array([r['root_zero'] for r in rr])),
                native_jvp_over_learned=A.grouped_ratio(rr,np.array([r['root_full'] for r in rr]),np.array([r['learned_'+k] for r in rr]))) for k in learned_models},
            hidden_ratios={k:A.grouped_ratio(rr,np.array([r['hidden_full'] for r in rr]),np.array([r['hidden_'+k] for r in rr])) for k in ['zero','residual']}))
    normsummary=[]
    for arm in ['random_norm','radial_norm']:
        for branch in sorted({r['branch'] for r in rows}):
            controls=[r for r in s['norm_controls'] if r['arm']==arm and rowby[r['id']]['branch']==branch]
            normsummary.append(dict(arm=arm,branch=branch,passed=sum(r['passed'] for r in controls),n=len(controls),max_relative_error=max(r['relative_error'] for r in controls)))
    unit_summary=[]
    for branch in sorted({r['branch'] for r in rows}):
        uu=[r for r in unit_rows if rowby[r['id']]['branch']==branch];values=np.array([r['smooth_zero_margin_contributions'] for r in uu]);rr=[rowby[r['id']] for r in uu]
        for j,u in enumerate(B.group()):unit_summary.append(dict(branch=branch,unit=u,mean_contribution=float(values[:,j].mean()),family_statistics=inference(family_values(rr,values[:,j]))))
    binary=None
    if split=='confirm':
        from scipy.stats import binomtest
        rr=[r for r in rows if r['depth']==1 and r['mode']=='archive' and r['title']==0]
        def correct(r,arm):
            ee=effect[r['id'],arm];return ee['scores'][r['gold']]>max(v for k,v in enumerate(ee['scores']) if k!=r['gold'])
        lost=sum(correct(r,'baseline') and not correct(r,'zero') for r in rr);gained=sum(not correct(r,'baseline') and correct(r,'zero') for r in rr)
        binary=dict(n_families=len(rr),lost=lost,gained=gained,accuracy_change=(gained-lost)/len(rr) if rr else None,
            p_exact=float(binomtest(lost,lost+gained,.5).pvalue) if lost+gained else 1.,full_power_scope_met=len(rr)==160)
    B.save(f'{split}-tail-report.json',dict(kind='real',summaries=summaries,primary_matched=paired,paired_binary=binary,complexity=complexity,title_control=title_control,path_clamps=paths,
        counterfactual_prediction=prediction,norm_control_qualification=normsummary,unit_linearized_attribution=unit_summary,metrics=metrics,gate_value_decomposition=nonlinear,
        numerical_checks=s['checks'],source_sha256=B.file_sha(__file__),
        limits=['Family averages, not intervention arms, are independent units.','Path clamps and root restoration are scoped manipulations, not unique mediation or proof of shared semantics.',
        'Smooth single-unit contributions use the baseline local Jacobian and frozen baseline competitor; no separate single-unit causal claim.',
        'RMS gate/value/cross terms do not sum as percentages and may cancel. Native BF16 rounding remainder is retained.',
        'A stronger depth effect alone is insufficient evidence of a new complex feature.']))
    B.log('tail_analysis_'+split,len(rows),len(rows),start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='confirm',choices=['confirm','format']);a=p.parse_args();summarize(a.split)
