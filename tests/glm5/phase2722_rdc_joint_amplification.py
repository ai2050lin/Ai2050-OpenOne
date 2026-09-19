"""Exploratory all-token amplitude events, full-coordinate forecasting and error accounting."""
from collections import Counter,defaultdict
import unicodedata
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
from rdc_joint_common import *
from rdc_joint_capture import ledger

OUT=BASE/'extension'


def token_kind(text):
    s=text.replace('Ġ',' ').replace('Ċ','\n').strip()
    if not s:return 'whitespace'
    if all(unicodedata.category(c).startswith('P') for c in s):return 'punctuation'
    if all(c.isnumeric() for c in s):return 'numeric'
    if any(c.isnumeric() for c in s):return 'mixed_numeric'
    if '\ufffd' in s:return 'multibyte_piece'
    return 'other'


def forecast(x,meta,event):
    tr=np.array([i for i,m in enumerate(meta) if m['split']=='train' and m['position']>0]);va=np.array([i for i,m in enumerate(meta) if m['split']=='validation' and m['position']>0])
    mu=x[tr].mean(0);sd=np.maximum(x[tr].std(0),1e-4);xx=((x-mu)/sd).astype(float)
    y=event.astype(float);a=xx[tr];b=y[tr];base=float(b.mean())
    all_predictions={'training_constant':np.full(len(y),base)};fits={};grid=[]
    # Native H12 is genuinely available before target H23; target ratios/events are never features.
    for penalty in (.0001,.01,1):
        def loss(theta):
            z=a@theta[:-1]+theta[-1];p=expit(z);err=p-b
            return float(np.mean(np.logaddexp(0,z)-b*z)+penalty*np.dot(theta[:-1],theta[:-1])/2),np.r_[a.T@err/len(b)+penalty*theta[:-1],err.mean()]
        initial=np.zeros(x.shape[1]+1);initial[-1]=np.log(base/(1-base))
        fit=minimize(loss,initial,jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-9,'gtol':1e-6,'maxcor':8})
        p=expit(xx@fit.x[:-1]+fit.x[-1]);v=float(np.mean(np.logaddexp(0,xx[va]@fit.x[:-1]+fit.x[-1])-y[va]*(xx[va]@fit.x[:-1]+fit.x[-1])))
        grid.append({'penalty':penalty,'validation_logloss':v,'iterations':int(fit.nit),'optimizer_converged':bool(fit.success),'message':str(fit.message)})
        fits[penalty]=(fit.x,p)
    selected=min(grid,key=lambda r:r['validation_logloss'])['penalty'];theta,p=fits[selected];all_predictions['full_H12_linear_logistic']=p
    npz(OUT/'event_forecast.npz',mean=mu,standard_deviation=sd,coefficient=theta[:-1],intercept=theta[-1:])
    counts=defaultdict(lambda:[0,0]);position=defaultdict(lambda:[0,0])
    for i in tr:
        m=meta[i];counts[m['token_id']][0]+=1;counts[m['token_id']][1]+=int(event[i])
        key=(m['language'],min(m['position']//8,7));position[key][0]+=1;position[key][1]+=int(event[i])
    def estimate(table,key):
        n,k=table[key];return (k+10*base)/(n+10)
    all_predictions['token_ID_shrink10']=np.array([estimate(counts,m['token_id']) for m in meta])
    all_predictions['language_position_shrink10']=np.array([estimate(position,(m['language'],min(m['position']//8,7))) for m in meta])
    results={}
    for split in ('validation','test','confirmation'):
        ii=np.array([i for i,m in enumerate(meta) if m['split']==split and m['position']>0]);yy=y[ii]
        results[split]={}
        for name,p in all_predictions.items():
            pp=p[ii];loss=-yy*np.log(np.maximum(pp,1e-15))-(1-yy)*np.log(np.maximum(1-pp,1e-15))
            results[split][name]={'tokens':len(ii),'events':int(yy.sum()),'average_precision':float(average_precision_score(yy,pp)) if yy.sum()>0 else None,
                'ROC_AUC':float(roc_auc_score(yy,pp)) if 0<yy.sum()<len(yy) else None,'Brier':float(np.mean((pp-yy)**2)),
                'logloss':float(loss.mean()),'group_logloss':paired_summary(loss,[meta[i]['source_group'] for i in ii])}
    npz(OUT/'event_forecast_all_token_predictions.npz',**{k:v.astype(np.float32) for k,v in all_predictions.items()})
    save(OUT/'event_forecast.json',{'timestamp':stamp(),'selected_penalty':selected,'grid':grid,'results':results,
        'algorithm':'Full2560 coordinate standardized H12 -> logistic ridge, train only; validation chooses penalty. Fixed token-ID and language/absolute-position shrinkage controls.',
        'limits':'Labels are numerical amplification events, not semantics. This extension follows inspection of main/fresh aggregate outcomes; reused confirmation is exploratory held-out evaluation, NOT a new blind replication. Full H12 already encodes past context.'})
    return results


def main():
    if (OUT/'amplification.json').exists():return
    start=time.monotonic();guard(35*1024**2)
    immutable(OUT/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'reason':'Follow rare internal H23 amplitudes and KL/MSE mismatch under the same authorized atlas objective.',
        'event':'Noninitial H23 energy > max(10,99.5th percentile TRAIN noninitial H23 energy) AND H23/H12 energy >=10. Full coordinate mean square, no coordinate or sample deletion.',
        'coverage':'Every native token of all768 frozen sources; first positions separately retained. Coordinate means/energies conditioned on numerical events, tokenizer categories and sample groups.',
        'prediction':'Full H12 linear logistic with L2 penalties .0001,.01,1, max150 L-BFGS iterations. Train only, validation logloss selects; constant, tokenID and language-position controls.',
        'accounting':'Decompose current H23/H36 full-coordinate squared error by amplitude event; all original primary errors remain unchanged.',
        'next_trace':'Up to64 deterministic event tokens (32train32confirmation) stratified language; pair same-token-ID non-event controls where available, unique source runtime replay once. All37layer x all2560 coordinates at selected tokens; native contributions at TRAIN-chosen maximum amplification block.',
        'calibration':'Separate full-vocabulary temperature control for frozen current_linear, temporal_embedding_bilinear and temporal_history_trilinear; original MSE/KL coefficients unchanged. Temperature train-fitted with validation step selection; same fixed routes evaluated on test/reused confirmation.',
        'epistemic_status':'New exploratory extension motivated after prior confirmation outputs. It is not fresh preregistered confirmation, and no retuning of original frozen routes.',
        'resource':'Within existing3GiB result/4h compute budget; no new parallel CUDA models or old data cleanup.'})
    material=rows()+rows(True);meta=[];energy=[];x=[];slices={}
    for ix,r in enumerate(material):
        z=field(r,r['split']=='confirmation');h=[unbits(z[k]) for k in ('h12','h23','h36')];ee=np.stack([np.mean(v.astype(float)**2,1) for v in h],1)
        begin=len(meta)
        for p in range(len(ee)):
            meta.append({k:r[k] for k in ('sample_id','source_group','split','language','genre')}|{'position':p,'token_id':r['prompt_ids'][p],
                'token':r['tokens'][p],'character_span':r['token_offsets'][p],'token_kind':token_kind(r['tokens'][p]),'language_mode_families':r['language_mode_families']})
        slices[r['sample_id']]=(begin,len(meta));energy.append(ee);x.append(h[0]);del z,h
        if (ix+1)%128==0:print('AMPLIFICATION_INVENTORY',ix+1,len(material),flush=True)
    energy=np.concatenate(energy);x=np.concatenate(x);training=np.array([m['split']=='train' and m['position']>0 for m in meta]);noninitial=np.array([m['position']>0 for m in meta])
    threshold=max(10.,float(np.quantile(energy[training,1],.995)));ratio=energy[:,1]/np.maximum(energy[:,0],1e-20);event=(energy[:,1]>threshold)&(ratio>=10)&noninitial
    save(OUT/'event_threshold.json',{'timestamp':stamp(),'training_noninitial_tokens':int(training.sum()),'H23_energy_threshold':threshold,'energy_ratio_min':10,'training_events':int(event[training].sum()),'definition_before_evaluation':True})
    counts={};eventrows=[];moments=defaultdict(lambda:np.zeros((2,3,2560),float));momentcounts=Counter()
    for r in material:
        a,b=slices[r['sample_id']];z=field(r,r['split']=='confirmation');h=np.stack([unbits(z[k]).astype(float) for k in ('h12','h23','h36')],1)
        for name,mask in [('event',event[a:b]),('non_event',noninitial[a:b]&~event[a:b]),('first',~noninitial[a:b])]:
            key=r['split']+'_'+name;v=h[mask];moments[key][0]+=v.sum(0);moments[key][1]+=(v*v).sum(0);momentcounts[key]+=len(v)
        del z,h
    npz(OUT/'all_token_event_coordinate_moments.npz',**{key:(v/max(momentcounts[key],1)).astype(np.float32) for key,v in moments.items()})
    for split in ('train','validation','test','confirmation'):
        mask=np.array([m['split']==split for m in meta]);nn=mask&noninitial;ee=mask&event
        counts[split]={'all_tokens':int(mask.sum()),'noninitial_tokens':int(nn.sum()),'events':int(ee.sum()),'event_fraction':float(event[nn].mean()),
            'groups_with_events':len({meta[i]['source_group'] for i in np.flatnonzero(ee)}),
            'event_share_noninitial_H23_squared_energy':float(energy[ee,1].sum()/energy[nn,1].sum()),
            'H23_energy_quantiles_noninitial':np.quantile(energy[nn,1],[0,.5,.9,.99,.995,1]).tolist(),
            'by_language':{l:{'tokens':int(sum(nn[i] and m['language']==l for i,m in enumerate(meta))),'events':int(sum(ee[i] and m['language']==l for i,m in enumerate(meta)))} for l in ('en','zh')},
            'event_token_kind':dict(Counter(meta[i]['token_kind'] for i in np.flatnonzero(ee)))}
    for i in np.flatnonzero(event):eventrows.append({**meta[i],'global_index':int(i),'energy_H12_H23_H36':energy[i].tolist(),'H23_H12_ratio':float(ratio[i])})
    allrows=[{**m,'energy_H12_H23_H36':energy[i].tolist(),'event':bool(event[i])} for i,m in enumerate(meta)]
    compressed_json(OUT/'all_token_amplitudes.json.gz',allrows);compressed_json(OUT/'event_tokens.json.gz',eventrows)
    npz(OUT/'all_token_energy.npz',energy=energy.astype(np.float32),event=event,position=np.array([m['position'] for m in meta]))
    predicted=forecast(x,meta,event);del x
    accounting={};anchor_rows=[]
    for split,folder in [('validation','main'),('test','main'),('confirmation','fresh')]:
        mm=read(BASE/'features'/folder/'rows.json');ii=[i for i,m in enumerate(mm) if m['split']==split];m=[mm[i] for i in ii]
        with np.load(BASE/'features'/folder/'targets.npz') as z:target=z['value'][ii,:5120]
        ev=np.array([event[slices[r['sample_id']][0]+r['position']] for r in m]);accounting[split]={}
        for name in ('current_linear','current_quadratic','rms_mean'):
            path=BASE/'confirmation/predictions'/f'current_{name}.npz' if folder=='fresh' else BASE/'rules/current'/name/'predictions.npz'
            with np.load(path) as z:p=z['prediction' if folder=='fresh' else split]
            err=(p.astype(float)-target)**2
            report={}
            for layer,u,v in [('H23',0,2560),('H36',2560,5120),('joint',0,5120)]:
                value=err[:,u:v].mean(1);report[layer]={'MSE':float(value.mean()),'events':int(ev.sum()),'anchors':len(ev),
                    'event_squared_error_fraction':float(value[ev].sum()/value.sum()),'event_MSE':float(value[ev].mean()) if ev.any() else None,
                    'non_event_MSE':float(value[~ev].mean()),'group_MSE':paired_summary(value,[r['source_group'] for r in m])}
            accounting[split][name]=report
            npz(OUT/f'{split}_{name}_event_error_coordinates.npz',event_error=err[ev].sum(0).astype(np.float32),non_event_error=err[~ev].sum(0).astype(np.float32))
        anchor_rows.extend([{**r,'event':bool(v)} for r,v in zip(m,ev)])
    compressed_json(OUT/'anchor_event_membership.json.gz',anchor_rows)
    selected=[]
    for split in ('train','confirmation'):
        chosen=[]
        for lang in ('en','zh'):
            candidates=[r for r in eventrows if r['split']==split and r['language']==lang]
            chosen.extend(candidates[:16])
        # Coverage can be asymmetric; deterministic fill, never discard all minority-language events.
        rest=[r for r in eventrows if r['split']==split and r not in chosen];chosen.extend(rest[:32-len(chosen)])
        for ev in chosen:
            match=next((m for i,m in enumerate(meta) if m['split']==split and m['token_id']==ev['token_id'] and m['position']>0 and not event[i]),None)
            selected.append({'event':ev,'same_token_non_event':match})
    immutable(OUT/'event_trace_selection.json',{'timestamp':stamp(),'pairs':selected,'selection':'First up to16 per language then remaining split order,32train32confirmation; same tokenID non-event earliest available in same split. No outcome retuning.'})
    result={'timestamp':stamp(),'all_sources':len(material),'all_native_tokens':len(meta),'threshold':threshold,'counts':counts,'moment_counts':dict(momentcounts),
        'state_error_accounting':accounting,'forecast':predicted,'selected_event_trace_pairs':len(selected),
        'limits':'All original samples retained. Numerical tail does not by itself identify semantics, causal gating, attention sink, or a new language family. Error decomposition is an identity conditional on event definition; classifier is fitted, not native factorization.'}
    save(OUT/'amplification.json',result);ledger('joint_all_token_amplification_inventory_and_full_coordinate_predictor',time.monotonic()-start,sources=len(material));guard()
    print('AMPLIFICATION_COMPLETE',json.dumps({'threshold':threshold,'counts':counts,'accounting':accounting},ensure_ascii=True),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
