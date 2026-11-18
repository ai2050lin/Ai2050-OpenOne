"""Read-only outcome reporting for frozen B1 repair; no fitting or threshold tuning."""
import json,sys,re
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import common_bench_b1_repair as R
B=R.B

def exposure():
    seen=set()
    for parent in ('common_bench_b1','common_bench_b1_v2'):
        for r in R.read(R.OLD.parent/parent/'pilot-items.json'):
            seen.update(r.get('entities',[]))
            if r.get('entity'):seen.add(r['entity'])
    return seen

def report_split(split):
    data=R.read(R.OUT/f'{split}-items.json');items=data['rows']
    byid={r['id']:r for r in R.read(R.OLD/f'{split}-materials.json')}
    rows=[byid[x['id']] for x in items]
    gate,mask=R.behavior_gate(rows,items)
    prim=[x for x,yes in zip(items,mask) if yes];rs=[r for r,yes in zip(rows,mask) if yes]
    ci=lambda v:R.rate_ci(v,rs)
    arms={}
    for name in data['arms']:
        a=[x['arms'][name] for x in prim]
        arms[name]={key:ci([v[key] for v in a]) for key in ['success','top1_success','unchanged','language_kept','other_language_target','kl_base_to_arm']}
        arms[name]['margin_change']=ci([v['margin']-v['base_margin'] for v in a])
        arms[name]['all_material_success']=float(np.mean([x['arms'][name]['success'] for x in items]))
        arms[name]['by_source']={str(s):R.rate_ci([v['success'] for v,r in zip(a,rs) if r['source']==s],[r for r in rs if r['source']==s]) for s in (1,-1)}
        if name.startswith('donor_'):
            arms[name]['donor_expected_success']=ci([v['pred']==v['donor_gold'] for v in a])
    norms={arm:{'valid_fraction':float(np.mean([x['arms'][arm]['relative_norm_error']<=.02 for x in items])),
                'max_relative_error':float(max(x['arms'][arm]['relative_norm_error'] for x in items))}
           for arm in ['unembed','random0','random1','random2','wrong_donor']}
    rate=lambda arm,key='success':arms[arm][key]['rate']
    controls={'norms':all(v['valid_fraction']>=.99 for v in norms.values()),
              'wrong_donor':rate('wrong_donor') is not None and rate('wrong_donor')<=.10}
    for arm in ['random0','random1','random2']:
        controls[arm]=rate(arm) is not None and rate(arm)<=.05 and rate(arm,'unchanged')>=.90
    calibration=bool(data['calibration']) and all(all(c.values()) for c in data['calibration'])
    rates={arm:rate(arm) for arm in ['pooled','cross_language','cross_source']}
    rates['cross_language_wrong_language']=rate('cross_language','other_language_target')
    verdict=R.decision(gate['passed'],all(controls.values()),calibration,rates)
    seen=exposure();exp={}
    for name,predicate in [('any_entity_seen',lambda r:bool(set(r['entities'])&seen)),('no_entity_seen',lambda r:not bool(set(r['entities'])&seen))]:
        selected=[(r,x) for r,x in zip(rs,prim) if predicate(r)]
        exp[name]={'n':len(selected),'arms':{a:R.rate_ci([x['arms'][a]['success'] for r,x in selected],[r for r,x in selected]) for a in ['pooled','cross_language','cross_source','unembed']}}
    compare={metric:ci([x['arms']['pooled'][metric]-x['arms']['unembed'][metric] for x in prim]) for metric in ['success','top1_success','kl_base_to_arm','margin']}
    paths={}
    for arm in R.PATH_ARMS:
        if arm not in data['arms']:continue
        a=[x['arms'][arm] for x in prim]
        actual=np.array([v['margin']-v['base_margin'] for v in a])
        predicted=np.array([v['prediction']['margin']-v['base_margin'] for v in a])
        denom=float(np.mean(actual**2));err=float(np.mean((actual-predicted)**2))
        paths[arm]={'n':len(a),'actual_mean_margin_change':float(actual.mean()),'predicted_mean_margin_change':float(predicted.mean()),
                    'zero_change_MSE':denom,'predicted_MSE':err,'MSE_reduction':1-err/denom if denom>0 else None,
                    'numerical_prediction_threshold_met':bool(denom>0 and err<=.9*denom),
                    'prediction_gate':bool(gate['passed'] and calibration and all(controls.values()) and denom>0 and err<=.9*denom),
                    'status':'behavior_ineligible' if not gate['passed'] else 'descriptive_controls_or_execution_failed' if not (calibration and all(controls.values())) else 'eligible_local_prediction',
                    'all_vocab_actual_KL':rate(arm,'kl_base_to_arm'),
                    'unchanged':rate(arm,'unchanged'),
                    'by_source':{str(s):{'n':int(sum(r['source']==s for r in rs)),
                       'actual_mean_margin_change':float(actual[np.array([r['source']==s for r in rs])].mean()),
                       'predicted_mean_margin_change':float(predicted[np.array([r['source']==s for r in rs])].mean())} for s in (1,-1)}}
        paths[arm]['paired_squared_error_improvement']=ci(actual**2-(actual-predicted)**2)
        paths[arm]['predicted_actual_correlation']=float(np.corrcoef(actual,predicted)[0,1])
        paths[arm]['centered_MSE_reduction_descriptive']=float(1-np.mean(((actual-actual.mean())-(predicted-predicted.mean()))**2)/np.var(actual)) if np.var(actual)>0 else None
        paths[arm]['centered_metric_caveat']='Post-hoc diagnostic, removes the observed panel mean from both. Not a deployable predictor or preregistered confirmation metric.'
    out={'split':split,'n':len(items),'primary_n':len(prim),'gate':gate,'calibration_passed':calibration,
         'controls':controls,'norm_checks':norms,'decisions':verdict,'arms':arms,'pooled_minus_unembed':compare,
         'path_predictions':paths,'exposure':{'prior_seen_entities':sorted({e for r in rows for e in r['entities']} & seen),**exp},
         'elapsed_seconds':data['elapsed_seconds'],'items_sha256':R.sha(R.OUT/f'{split}-items.json'),
         'status':'limited historical-candidate replication; no automatic formal promotion'}
    R.save(f'{split}-report.json',out)
    print(json.dumps({k:out[k] for k in ['split','n','primary_n','gate','controls','decisions','path_predictions']},ensure_ascii=False),flush=True)
    return out

def generation():
    data=R.read(R.OUT/'generation.json')['rows'];result={}
    for split in ('lockbox','lockbox_frame'):
        materials={r['id']:r for r in R.read(R.OLD/f'{split}-materials.json')}
        for arm in ['base','pooled','unembed','cross_language']:
            rows=[r for r in data if r['split']==split and r['arm']==arm];hit=[];strict=[];examples=[]
            for r in rows:
                mat=materials[r['id']];color=mat['gold_color'] if arm=='base' else B.COLORS[r['target_color']]
                ans=color if mat['language']==1 else B.ZH[color]
                text=r['text'].strip().strip('"\'')
                matched=bool(re.match(re.escape(ans)+(r'\b' if mat['language']==1 else r'(?:色)?'),text,re.I))
                exact=text.lower().strip(' .。!！\n') in ([ans] if mat['language']==1 else [ans,ans+'色'])
                hit.append(matched);strict.append(exact)
                if not matched and len(examples)<6:examples.append({**r,'expected':ans})
            result[f'{split}/{arm}']={'leading_answer':R.rate_ci(hit,[materials[r['id']] for r in rows]),
                'whole_answer_exact':R.rate_ci(strict,[materials[r['id']] for r in rows]),'failures':examples}
    return result

def capability():
    data=R.read(R.OUT/'capability.json')['rows'];out={}
    base=np.array([r['arms']['base']['correct'] for r in data],float)
    for arm in ['base','pooled','unembed','random0']:
        acc=np.array([r['arms'][arm]['correct'] for r in data],float);delta=acc-base
        rng=R.rng_for(arm,'cap-ci');boot=np.mean(delta[rng.integers(len(data),size=(5000,len(data)))],1)
        bounds=np.quantile(boot,[.025,.975]).tolist()
        out[arm]={'n':len(data),'accuracy':float(acc.mean()),'accuracy_change':float(delta.mean()),'paired_item_ci95':bounds,
            'mean_full_vocab_KL':float(np.mean([r['arms'][arm]['kl'] for r in data])),
            'retention':'supported_on_this_pool' if bounds[0]>=-.02 else 'failed' if bounds[1]<-.02 else 'inconclusive',
            'by_subject':{s:{'n':sum(r['subject']==s for r in data),'accuracy':float(np.mean([r['arms'][arm]['correct'] for r in data if r['subject']==s]))} for s in sorted({r['subject'] for r in data})},
            'caveat':'Exposed convenience MCQ pool; item bootstrap is conditional, not broad ability guarantee.'}
    return out

def main():
    R.checked();reports={}
    for split in ('discovery','lockbox','lockbox_frame'):
        if (R.OUT/f'{split}-items.json').exists():reports[split]=report_split(split)
    extras={}
    if (R.OUT/'generation.json').exists():extras['generation']=generation()
    if (R.OUT/'capability.json').exists():extras['capability']=capability()
    both=all(s in reports and reports[s]['gate']['passed'] and reports[s]['calibration_passed'] and all(reports[s]['controls'].values()) for s in ['lockbox','lockbox_frame'])
    R.save('report.json',{'splits':reports,**extras,'formal_evidence_promotion':0,
        'both_panel_admissibility':both,'cross_frame_mechanism_confirmation':'not_admissible' if not both else 'evaluate_frozen_path_thresholds',
        'limitations':['Two lockboxes share entity roster and some historical pilot exposure.',
                       'State shifts are not naturally occurring computations; head ablations are restricted to the final query.',
                       'Prediction uses real internal source states, not autonomous text-to-output inference.']})

if __name__=='__main__':main()
