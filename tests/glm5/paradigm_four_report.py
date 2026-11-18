"""Analyze frozen margins, prediction, and native path comparisons, then append the memo."""

from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json

import numpy as np

import paradigm_four_run as X
import paradigm_four_discovery as D
import paradigm_four_material as M
import paradigm_routes_report as old
import record_update_report as memo
import record_update_study as ridge
from deploy.storage_policy import check_write


R,C,OUT=X.R,X.C,X.OUT
TITLE='四类语言范式的原生主干筛选、融合竞争与随机单元审计'
AXES=X.FACTORS
LEVELS=np.asarray(M.LEVELS)


def summary(values):
    return old.family_summary(np.asarray(values))


def restore_bits(arr):
    return old.unbits(arr)


def data_load(files):
    values={}
    for file in files:
        path=OUT/file
        binding=R.read(path.with_suffix('.json').name)
        assert binding['sha']==C.A.digest(path) and binding['plan']==C.A.digest(OUT/'plan.json')
        with np.load(path,allow_pickle=False) as z:
            for key in z.files:
                values.setdefault(key,[]).append(z[key])
    return {k:np.concatenate(v) for k,v in values.items()}


def behavior(data,rows,tok):
    margin=data['margin'].reshape(-1,16,4)
    signed=data['margin']*(2*np.asarray([row['dims'] for row in rows])-1)
    result={}
    for axis,key in enumerate(AXES):
        flag=8>>axis
        pairs=[(i,i+flag) for i in range(16) if not i&flag]
        gaps=np.stack([margin[:,hi,axis]-margin[:,lo,axis] for lo,hi in pairs],-1)
        strata=(gaps>0).mean(0)
        result[key]=dict(paired=summarize_family(gaps),minimum_stratum=float(strata.min()),
                         fraction_by_stratum=strata.tolist(),absolute=float((signed[:,axis]>0).mean()),
                         passed=bool((strata>=.75).all()))
    starts=tok.batch_decode([[int(k)] for k in data['top'][:,0]])
    expected=[row['dims'][1]==(word=='print') for row,word in zip(rows,starts,strict=True)]
    for outcome in (0,1):
        keep=np.asarray([row['dims'][1]==outcome for row in rows])
        result['format'][f'first_token_hit_{outcome}']=float(np.asarray(expected)[keep].mean())
    result['format']['non_print_interpretation']='Academic side counts every non-print top1; it is not a complete academic-style classifier.'
    return result


def summarize_family(gaps):
    return summary(gaps.mean(1))


def factorial(data):
    full=data['fusion']
    stats={}
    for method,i in [('base',0),('additive',1),('pairwise',2)]:
        stats[method]=dict(kl=summary(full[:,:,i,0].mean(1)),top1=summary(full[:,:,i,1].mean(1)))
    improvement=summary((full[:,:,1,0]-full[:,:,2,0]).mean(1))
    improvement['relative']=improvement['mean']/max(stats['additive']['kl']['mean'],1e-10)
    improvement['passed']=improvement['ci'][0]>0 and improvement['relative']>=.10
    stats['gain']=improvement
    stats['high_order_cells']=[M.LEVELS[i] for i in X.COMBINED]
    stats['per_cell_KL']=full[:,:,:,0].mean(0).tolist()
    return stats


def covariates(rows,data):
    base=[]
    for r in rows:
        levels=np.asarray(r['dims'])
        word_features=np.bincount(np.asarray(r['token']['ids'][:r['query'][0]+1])%32,minlength=32)
        base.append(np.r_[levels,*[levels[i]*levels[j] for i,j in combinations(range(4),2)],
                          r['archive'],r['gold']/9.,r['query'][0]/160,word_features/len(r['token']['ids'])])
    act=restore_bits(data['gua'][:,0,:,2])
    mass=data['mass'][:,0].reshape(len(rows),-1)
    head=np.asarray(base,np.float32)
    return dict(text=head,source=np.c_[head,mass],native=np.c_[head,mass,act[:,:32]],random=np.c_[head,mass,act[:,32:]])


def fit_pilot(rows,data):
    file=OUT/'pilot-predictor.json'
    if file.is_file():
        existing=R.read(file.name)
        assert existing['fit']==C.A.digest(OUT/'pilot-predictor.npz')
        return
    fits={}
    for name,inputs in covariates(rows,data).items():
        for key,value in ridge.linear(inputs,data['margin'],12.).items():
            fits[f'{name}.{key}']=value
    digest=X.base.write_np('pilot-predictor.npz',fits)
    R.write('pilot-predictor.json',dict(at=C.stamp(),fit=digest,plan=C.A.digest(OUT/'plan.json'),
            method='Ridge-frozen pilot forecasts matched probe margins. Data at shared onset only. Equal count native/random B30 units, same group-mass controls. All axes distinct units and scales.'))


def forecast_margin(rows,data):
    manifest=R.read('pilot-predictor.json')
    assert manifest['fit']==C.A.digest(OUT/'pilot-predictor.npz')
    fitted=X.base.read_np('pilot-predictor.npz')
    report={};errors={}
    for name,inputs in covariates(rows,data).items():
        fit={k.split('.',1)[1]:v for k,v in fitted.items() if k.startswith(name+'.')}
        forecast=ridge.evaluate(fit,inputs)
        errors[name]=(forecast-data['margin'])**2
        report[name]={axis:summary(errors[name][:,i].reshape(-1,16).mean(1)) for i,axis in enumerate(AXES)}
    report['random_minus_native']={axis:summary((errors['random'][:,i]-errors['native'][:,i]).reshape(-1,16).mean(1))
                                    for i,axis in enumerate(AXES)}
    return report


def natural_path(data,candidate):
    fields=restore_bits(data['gua'][:,0]).reshape(-1,16,64,3)
    mass=data['mass'][:,0].reshape(-1,16,2,len(M.REGIONS))
    ratio=np.log(np.maximum(mass[:,:,:,M.REGIONS.index('format')],1e-8)/np.maximum(mass[:,:,:,M.REGIONS.index('archive')],1e-8))
    result={}
    strength={}
    for axis,key in enumerate(AXES):
        low=LEVELS[:,axis]==0
        contrast=fields[:,~low].mean(1)-fields[:,low].mean(1)
        values=ratio[:,~low].mean(1)-ratio[:,low].mean(1)
        result[key]=[summary(values[:,i]) for i in range(2)]
        strength[key]=dict(selected_abs_mean=np.abs(contrast[:,:32]).mean(axis=(0,1)).tolist(),
                           random_abs_mean=np.abs(contrast[:,32:]).mean(axis=(0,1)).tolist())
    top=int(candidate['selected'][0])
    return dict(heads=candidate['heads'],baseline='request/reference QK log mass ratio at common onset',
                ratios=result,gua=strength,first_unit=top,
                first_unit_mean_GUA_by_cell=fields[:,:,0].mean(0).tolist())


def causal(rows,data,plan,tok):
    n=len(data['edit_top']);rows=rows[:n]
    groups=np.asarray([row['archive']!=row['dims'][1] for row in rows]).reshape(-1,16)
    sign=np.asarray([2*row['dims'][1]-1 for row in rows])
    edit=data['edit_margin'];top=data['edit_top']
    raw_aligned=top[:,0]==data['top'][:n,0]
    aligned=raw_aligned.reshape(-1,16).all(1)
    cohort=np.where(aligned)[0]
    keep=aligned.repeat(16)
    dose=data['mlp_doses']
    pair_effect=edit[:,4]-edit[:,2]-edit[:,3]+edit[:,0]
    def score(value,mask):
        grouped=value.reshape(-1,16)
        vals=[grouped[i,mask[i]].mean() for i in cohort]
        return summary(vals) if vals else dict(n=0,mean=None,ci=None)
    archive_removed=score((edit[:,2]-edit[:,0])*sign,groups)
    request_removed=score((edit[:,0]-edit[:,3])*sign,groups)
    interaction=score(pair_effect*sign,groups)
    relative=np.abs(dose[:,0]-dose[:,1])/np.maximum(np.maximum(dose[:,0],dose[:,1]),.02)
    p95=float(np.quantile(relative[keep],.95)) if keep.any() else None
    receivers=data['native_gua'][:,4]-data['native_gua'][:,0]
    rescue=(edit[:,6]-edit[:,7])*sign
    selected_effect=score(rescue,groups)
    format_gate=R.read('pilot-summary.json')['behavior']['format']['passed']
    # Preserve actual counterexamples, including reversed native effects.
    order=np.argsort((edit[:,2]-edit[:,0])*sign)
    cases=[]
    for j in order[:8]:
        row=rows[j]
        cases.append(dict(id=row['id'],dims=row['dims'],archive=row['archive'],sham_aligned=bool(raw_aligned[j]),
                          margins=edit[j].tolist(),top=tok.batch_decode([[int(k)] for k in top[j]])))
    return dict(families=int(aligned.sum()),total_families=n//16,sham_views=int(raw_aligned.sum()),
                format_pilot_passed=format_gate,archive_removed_toward_requested=archive_removed,
                request_removal_loss=request_removed,joint_nonadditivity=interaction,
                functional_competition_passed=bool(format_gate and aligned.sum()>=60 and
                    all(v['mean']>=.10 and v['ci'][0]>0 for v in (archive_removed,request_removed))),
                counts=dict(zip(X.ARMS,(np.sum((top!=top[:,[0]])&keep[:,None],axis=0)).tolist())),
                native_full_restore_equal=bool(np.array_equal(edit[:,0],edit[:,5]) and np.array_equal(top[:,0],top[:,5])),
                receiver_abs_GUA_change=dict(selected=np.abs(receivers[keep,:32]).mean(axis=(0,1)).tolist() if keep.any() else None,
                                             random=np.abs(receivers[keep,32:]).mean(axis=(0,1)).tolist() if keep.any() else None),
                selected_rescue_vs_random=selected_effect,rescue_dose_p95=p95,matched_rescue=p95 is not None and p95<=.03,
                excluded_families=[rows[16*j]['family'] for j in range(n//16) if not aligned[j]],counterexamples=cases)


def analyze(rows,parts,plan,tok,split):
    data=data_load(parts)
    assert list(data['ids'])==[row['id'] for row in rows]
    behavior_stats=behavior(data,rows,tok)
    record=dict(at=C.stamp(),plan=C.A.digest(OUT/'plan.json'),parts={p:C.A.digest(OUT/p) for p in parts},
                behavior=behavior_stats,fusion=factorial(data))
    if split=='pilot':
        fit_pilot(rows,data)
    else:
        # Demonstrate identical onset embedding across the explicitly audited family views.
        field=data['field'][:,0,0]
        for group in field.reshape(-1,16,5120):
            assert np.array_equal(group,np.repeat(group[[0]],16,axis=0))
        record.update(forecast=forecast_margin(rows,data),path=natural_path(data,R.read('candidate.json')),
                      causal=causal(rows,data,plan,tok),field_audit=dict(views=len(field),all_shared_embedding_equal=True))
        high=data['fusion'][:,:,2,0]-data['fusion'][:,:,1,0]
        worst=np.unravel_index(np.argsort(-high.ravel())[:12],high.shape)
        record['fusion_counterexamples']=[dict(id=rows[int(f)*16+X.COMBINED[int(c)]]['id'],
             additive_KL=float(data['fusion'][f,c,1,0]),pairwise_KL=float(data['fusion'][f,c,2,0])) for f,c in zip(*worst)]
    R.write(split+'-summary.json',record)
    print(split,'behavior',[(k,x['passed']) for k,x in behavior_stats.items()],
          'fusion',record['fusion']['gain'],flush=True)


def show(stat):
    return memo.show(stat)
