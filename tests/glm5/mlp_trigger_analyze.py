"""Score the frozen trigger/shared-input protocol, retaining every failure."""
from collections import Counter
import numpy as np
from scipy.stats import binomtest,spearmanr,beta
import mlp_trigger_design as D
import mlp_trigger_run as R

def corr(x,y):
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])

def interval(values,alpha=.005):
    x=np.asarray(values);rng=np.random.default_rng(D.SEED)
    a=np.concatenate([x[rng.integers(0,len(x),(250,len(x)))].mean(1) for _ in range(20)])
    return np.quantile(a,[alpha/2,1-alpha/2]).tolist()

def mse_score(error,base):
    a=np.asarray(error).reshape(D.N_HOLD,8).mean(1);b=np.asarray(base).reshape(D.N_HOLD,8).mean(1)
    wins=int((a<b).sum());ratio=float(a.mean()/max(b.mean(),1e-30));pv=float(binomtest(wins,len(a),.5,alternative='greater').pvalue)
    return {'mse_ratio':ratio,'family_wins':wins,'families':len(a),'family_win_rate':wins/len(a),'p':pv,
            'gain_interval':interval(b-a),'passed':ratio<=.90 and wins/len(a)>=.65 and pv<=.005}

def main():
    p=D.checked();rows,r,y,nuis,meta=R.arrays('holdout',p);sel=D.READ(D.OUT/'selection.json');observations={};counterexamples=[]
    behavior={str(frame):float(np.mean([np.argmax(m['scores'][:2])==row['gold'] for row,m in zip(rows,meta) if row['frame']==frame])) for frame in range(16)}
    for i,key in enumerate(D.PAIRS):
        models=sel['predictors'][key];idx=sel['coordinates'][key]
        baseline=R.predict(models['nuisance'],nuis[:,i]);source=R.predict(models['source'],np.column_stack([nuis[:,i],y[:,i,0]]))[:,0]
        common=R.predict(models['common'],np.column_stack([nuis[:,i],r[:,i,idx]]))
        source_error=(source-y[:,i,1])**2;common_error=(common[:,1]-y[:,i,1])**2
        err=common-y[:,i];ratio=mse_score(common_error,(baseline[:,1]-y[:,i,1])**2)
        observations[key]={'coordinates':idx,'pearson':corr(y[:,i,0],y[:,i,1]),
          'spearman':float(spearmanr(y[:,i,0],y[:,i,1]).statistic),'winsorized':corr(*[np.clip(y[:,i,k],*np.quantile(y[:,i,k],[.01,.99])) for k in range(2)]),
          'common_input_prediction':ratio,'common_vs_source_only':mse_score(common_error,source_error),
          'source_prediction':mse_score((common[:,0]-y[:,i,0])**2,(baseline[:,0]-y[:,i,0])**2),
          'residual_correlation':corr(err[:,0],err[:,1]),'scope':'upstream R observed; different readouts: broad final, rare maximum coordinate4 after token63'}
        errors=common_error.reshape(D.N_HOLD,8).mean(1)
        for f in np.argsort(errors)[-4:][::-1]:
            k=int(f)*8+int(np.argmax(common_error.reshape(D.N_HOLD,8)[f]))
            counterexamples.append({'type':'prediction_error','pair':key,'family':int(f),'row':rows[k],'position':meta[k]['positions'][key],
                'actual':y[k,i].tolist(),'predicted':common[k].tolist(),'error':float(errors[f])})
    triggers=[];counts=np.array([len(m['rare_positions']) for m in meta]).reshape(D.N_HOLD,8)
    for long in range(2):
      for rep in range(2):
       for past in range(2):
        ii=[i for i,v in enumerate(rows) if (v['long'],v['repetition'],v['past'])==(long,rep,past)]
        peaks=[x for i in ii for x in meta[i]['rare_positions']];energy=[x for i in ii for x in meta[i]['rare_R4_energy']]
        triggers.append({'long':long,'repetition':rep,'past':past,'n':len(ii),
            'tokens_range':[min(len(rows[i]['tokens']) for i in ii),max(len(rows[i]['tokens']) for i in ii)],
            'any_event':sum(bool(meta[i]['rare_positions']) for i in ii),
            'mean_events':float(np.mean([len(meta[i]['rare_positions']) for i in ii])),
            'first600_mean_events':float(np.mean([meta[i]['rare_count_first_600'] for i in ii])),
            'peak_position_quantiles':np.quantile(peaks,[.05,.5,.95]).tolist() if peaks else [],
            'R4_energy_median':float(np.median(energy)) if energy else None})
    trigger_effects=[]
    for axis,label in [(0,'length'),(1,'repetition'),(2,'tense')]:
        grid=counts.reshape(D.N_HOLD,2,2,2);delta=(np.take(grid,1,axis=axis+1)-np.take(grid,0,axis=axis+1)).mean((1,2))
        trigger_effects.append({'factor':label,'mean_event_difference':float(delta.mean()),'interval':interval(delta),
          'scope':'event count, fixed material family; length intentionally changes exposure; first600 counts reported separately'})
    token_counts=Counter()
    for row,m in zip(rows,meta):
        for pos in m['rare_positions']:token_counts[row['tokens'][pos]]+=1
    causal=[D.READ(D.OUT/f'causal-{f:04d}.json') for f in range(D.N_HOLD)];curves=[]
    for key in D.PAIRS:
      for mode,dose in [('upstream',.25),('upstream',.5),('upstream',1.),('unit',1.)]:
        rr=[];pred=[]
        for f,doc in enumerate(causal):
            assert doc['prediction_sha256']==D.HASH(D.OUT/f'prediction-{f:04d}.json')
            rr.append(next(x for x in doc['records'] if (x['pair'],x['mode'],x['dose'])==(key,mode,dose)))
            pred.append(next(x for x in D.READ(D.OUT/f'prediction-{f:04d}.json')['predictions'] if (x['pair'],x['mode'],x['dose'])==(key,mode,dose)))
        actual=np.array([a['target_delta'] for a in rr]);control=np.array([a['control_target_delta'] for a in rr]);pr=np.array([a['target_delta'] for a in pred]);valid=np.array([a['norm_valid'] for a in rr])
        switches=sum(np.argmax(a['effect']['scores'][:2])!=np.argmax(c['baseline']['scores'][:2]) for a,c in zip(rr,causal))
        rec={'pair':key,'mode':mode,'dose':dose,'n':len(rr),'valid_controls':int(valid.sum()),
             'target_delta_mean':float(actual.mean()),'target_delta_interval':interval(actual),
             'control_delta_mean':float(control.mean()),'excess_abs_effect_interval':interval(abs(actual[valid])-abs(control[valid])) if valid.any() else None,
             'forecast_mse_vs_nochange':float(np.mean((actual-pr)**2)/max(np.mean(actual**2),1e-30)),
             'first_token_changes':sum(a['effect']['top1']!=c['baseline']['top1'] for a,c in zip(rr,causal)),
             'answer_switches':int(switches),'control_answer_switches':int(sum(np.argmax(a['control']['scores'][:2])!=np.argmax(c['baseline']['scores'][:2]) for a,c in zip(rr,causal))),
             'max_norm_relative_error':max(abs(a['norm']-a['control_norm'])/a['norm'] for a in rr if a['norm']),
             'mean_kl':float(np.mean([a['effect']['kl_to_baseline'] for a in rr])),
             'mean_forecast_kl':float(np.mean([a['effect']['kl_to_prediction'] for a in rr]))}
        if mode=='upstream' and dose==1:
            for clamp in ['unit','mlp']:
                c=np.array([a[clamp+'_clamp']['target_delta'] for a in rr])
                rec[clamp+'_clamp_retained_l1']=float(abs(c).sum()/max(abs(actual).sum(),1e-30))
                rec[clamp+'_clamp_difference_mean']=float((c-actual).mean())
            rec['bypass_candidate_in_scope']=bool(valid.mean()>=.95 and rec['unit_clamp_retained_l1']>=.8 and rec['excess_abs_effect_interval'][0]>0)
        curves.append(rec)
        for f in np.argsort(abs(actual-pr))[-2:]:
            counterexamples.append({'type':'intervention_error','pair':key,'mode':mode,'dose':dose,'family':int(f),
              'sample_id':causal[f]['sample_id'],'actual':float(actual[f]),'prediction':float(pr[f]),'control':float(control[f]),
              'outcome':rr[f]['effect'],'baseline':causal[f]['baseline']})
    cap=D.READ(D.OUT/'capability.json')['rows'];cap_summary={}
    for label in ['broad-upstream','rare-upstream','broad-unit','rare-unit']:
        lost=sum(np.argmax(v['scores']['baseline'])==v['gold'] and np.argmax(v['scores'][label])!=v['gold'] for v in cap)
        gained=sum(np.argmax(v['scores']['baseline'])!=v['gold'] and np.argmax(v['scores'][label])==v['gold'] for v in cap)
        upper=float(beta.ppf(1-.0125,lost+1,len(cap)-lost)) if lost<len(cap) else 1.
        cap_summary[label]={'n':len(cap),'baseline_correct':int(sum(np.argmax(v['scores']['baseline'])==v['gold'] for v in cap)),
                            'lost':int(lost),'gained':int(gained),'conservative_loss_upper':upper,'passed':upper<.02}
    report={'schema':'mlp_trigger_report.v1','run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'kind':'real','power':p['power'],
       'families':D.N_HOLD,'samples':8*D.N_HOLD,
       'behavior':{'per_frame':behavior,'passed':min(behavior.values())>=.85},'observations':observations,
       'triggers':triggers,'trigger_effects':trigger_effects,'event_token_types':len(token_counts),'top_event_tokens':token_counts.most_common(12),
       'dose_curve':curves,'capability':cap_summary,'counterexamples':counterexamples,
       'restoration_checks':sum(d['bitwise_restored'] for d in causal),'repeatability':D.READ(D.OUT/'repeatability.json'),
       'limits':p['limits']+['Readout argmax |R33[4]| is an upstream-conditioned selection, not prediction of its position from language alone.',
           'Capability at final token does not certify earlier-token interventions, full generation or other task distributions.',
           'Cancel/replay validates implementation; source and full-MLP clamps test local bypass only, not a unique head or complete computational mechanism.',
           'One physical node; independent replication and formal scientific audit pending.'],
       'formal_evidence_promotion':False}
    D.save('report.json',report,p)
    print({'behavior':report['behavior']['passed'],'triggers':triggers,'curves':curves,'capability':cap_summary},flush=True)

if __name__=='__main__':main()
