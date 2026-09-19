"""Predict later native attention from an earlier query and already available past KV.

No model is loaded. All native coordinates and every past cache position participate.
The current token's L23 K/V and Q are targets, never prediction inputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,gc,shutil
from scipy.special import softmax
from rdc_conditional_common import *
from rdc_conditional_estimators import RIDGES
from phase2704_rdc_predictive_gates import errors

OUT=CAMPAIGN/'n_cached_attention';M=CAMPAIGN/'m_order'
GROUPS=('record','prompt_other','generated_trace','generated_neutral','generated_result','generated_other')
MAX_PAST=512


def prepare():
    assert (M/'result.json').exists(),'Finish M scoring and analysis before this automatic extension'
    path=OUT/'protocol.json'
    protocol={'phase':2708,'source_sha':sha(Path(__file__)),'timestamp':stamp(),
      'same_goal':True,'reason':'M supplies actual full-source attention paths at a fixed output-field boundary; K/J show that early hidden prediction and same-block reconstruction must be separated. Test a new cache-conditional attention operator, not another donor patch or label probe.',
      'material':'All M score-v2 valid Result-field onsets, regardless of correctness; original entity0..3 train/4..5 validation/6..7 test. Missing-field cases remain M behavior failures but have no boundary state for this analysis. This reuses M, not an independent new material confirmation.',
      'available_inputs':'Current query H12; optionally every L23 past K/V coordinate at positions0..query_position-1. These earlier-token cache entries already exist before the current decoding step. Exclude current-token K/V, current L23 Q/P/attention_x, future H, emitted next token and answer labels. Query position and source-text labels are available from the current prefix. This is a conditional observed-boundary forecast, not prediction of when that boundary will occur.',
      'targets':'All32x128 saved L23 query coordinates, current8x128 K and V, and actual4096 head-output /2560 O-projection output. Q and current K are demodulated with a known-position two-half RoPE convention for fitting, then remodulated; this is an invertible coordinate convention for saved rounded values, not recovery of exact pre-rounding native Q/K or rotation of HiddenState.',
      'inputs_and_kernels':'H12-only and H12 plus full past K/V, each block RMS-scaled using training only, then linear/quadratic Gram. Cache uses fixed512 left-aligned past positions and zero padding; no past coordinate is reduced, ranked, or discarded. Cached states are temporary RAM, original M arrays retained. All input Grams saved for recomputation checks.',
      'estimators':'Identical Gram/ridgegrid for factor Q+Kself+Vself (6144 outputs), equal-width direct head-output+Kself+Vself (6144 outputs), and direct attention output (2560). All select ridge by validation native attention-output MSE. Direct extra K/V heads match output count but do not give identical useful supervision. Four input/kernel combinations. Also train-global and family/language/order mean factors with actual past cache, mean direct heads, and fixed effective-df64 factor/head controls for full-cache inputs. Groups without a valid training boundary use the global training mean, never heldout targets.',
      'composition':'Predicted Q, current K/V plus observed past K/V -> all-source32-head scaled dot softmax -> full head output -> real W_O. Optional head-norm calibration uses only training mean norm of position-demodulated Q/K; reported separately, not selected after test results. Native Q/K/V arithmetic floor is an oracle audit, not a forecast.',
      'outputs':'Per-case/per-coordinate attention-output errors, all head probabilities over every source (zero-padded only outside prefix), all six source-group2560-vectors, models, Gram matrices and source identities. No Top-K or latent-axis visualization.',
      'resources':{'maximum_seconds':1200,'maximum_new_bytes':768*1024**2,'campaign_ceiling':30*1024**3,'free_floor':8*1024**3,'additional_CUDA_models':0},
      'limits':['Only2heldoutentitygroups; conditioned on valid field labels.','Full past KV carries previous deeper computation, so this is not a replacement for the whole decoder or a pure H12-only model.','Query/head factorization may lose to direct prediction; neither outcome proves an entire paradigm impossible.','Not native answer correctness, causal mediation, necessity, or a uniquely identified semantic program.','Known position convention and source groups are bookkeeping, not an assumption that HiddenState itself follows RoPE.']}
    if path.exists():assert read(path)['source_sha']==protocol['source_sha'],'Protocol source changed; version explicitly before rerun'
    else:immutable(path,protocol)
    return read(path)


def rotation(position):
    # Native implementation pairs first/second 64 coordinates. This FP64 convention
    # deliberately does not claim to undo the original rounded intermediate products.
    angle=float(position)/(1000000.**(np.arange(0,128,2,dtype=np.float64)/128.))
    return np.cos(angle),np.sin(angle)


def rotate(x,position,inverse=False):
    c,s=rotation(position);a,b=x[...,:64],x[...,64:]
    return np.concatenate((a*c+b*s,-a*s+b*c),-1) if inverse else np.concatenate((a*c-b*s,a*s+b*c),-1)


def group_key(r):return (r['family'],r['language'],r['order'])


def collect():
    prefixes=read(M/'prefixes.json');rows=[];objects=[];h=[];qkv=[];heads=[];actual=[];oracle=[];links=[]
    for r in prefixes:
        b=read(M/f'behavior_scored/{r["sample_id"]}.json');sid=b['result_boundary_state']
        if sid is None:continue
        row=read(M/f'steps/{sid}.json');assert row['generation_step']>=1
        path=M/f'fields/{sid}.npz'
        with np.load(path) as z:
            q,k,v,p,head,a=[unbits(z['L23_'+part]).astype(np.float64) for part in ('q','k','v','p','head_output','attention_out')]
            hh=unbits(z['h_c'][12]).astype(np.float32)
        n=row['query_position'];assert k.shape==v.shape==(8,n+1,128) and q.shape==(32,128) and p.shape==(32,n+1) and n<=MAX_PAST
        assert len(row['source_labels'])==n+1 and set(row['source_labels']).issubset(GROUPS)
        qq=rotate(q,n,True);kk=rotate(k[:,-1],n,True)
        assert np.max(np.abs(rotate(qq,n)-q))<1e-10 and np.max(np.abs(rotate(kk,n)-k[:,-1]))<1e-10
        current=dict(row,result_correct=b['scores']['result_correct']);rows.append(current)
        objects.append({'past_k':k[:,:-1].astype(np.float32),'past_v':v[:,:-1].astype(np.float32),'native_p':p.astype(np.float32),
          'position':n,'groups':np.array([GROUPS.index(g) for g in row['source_labels']],int)})
        h.append(hh);qkv.append(np.concatenate([qq.reshape(-1),kk.reshape(-1),v[:,-1].reshape(-1)]));heads.append(head);actual.append(a)
        oracle.append((q,k,v));links.append({'sample_id':sid,'field_sha':sha(path),'behavior_scored_sha':sha(M/f'behavior_scored/{r["sample_id"]}.json')})
    assert len(rows)==read(M/'scoring_alignment_audit.json')['valid_result_field_onsets']
    data={'h12':np.stack(h),'qkv':np.stack(qkv),'head':np.stack(heads),'attention':np.stack(actual)}
    npz(OUT/'features.npz',**data);save(OUT/'selected_rows.json',rows);save(OUT/'source_links.json',links)
    return rows,objects,data,oracle


def cache_gram(objects,tr):
    n=len(objects);grams={};scales={}
    for name in ('past_k','past_v'):
        gram=np.zeros((n,n),np.float64)
        for start in range(0,MAX_PAST,32):
            block=np.zeros((n,32,1024),np.float64)
            for i,o in enumerate(objects):
                x=o[name][:,start:start+32].transpose(1,0,2).reshape(-1,1024);block[i,:len(x)]=x
            flat=block.reshape(n,-1);gram+=flat@flat.T
        scale=max(float(np.sqrt(np.mean(np.diag(gram)[tr]))),1e-12);grams[name]=gram/scale**2;scales[name]=scale
        print('CACHE_GRAM',name,n,flush=True)
    return grams,scales


def compose(factors,indices,objects,wo,norms=None,keep=False):
    head=[];probs=[];vectors=[];maxn=max(o['position']+1 for o in objects)
    for prediction,i in zip(factors,indices):
        o=objects[i];position=o['position'];q=prediction[:4096].reshape(32,128).astype(np.float64);ks=prediction[4096:5120].reshape(8,128).astype(np.float64);vs=prediction[5120:].reshape(8,128).astype(np.float64)
        if norms is not None:
            q=q*norms[0][:,None]/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
            ks=ks*norms[1][:,None]/np.maximum(np.linalg.norm(ks,axis=1,keepdims=True),1e-12)
        q=rotate(q,position);ks=rotate(ks,position)
        k=np.concatenate((o['past_k'].astype(np.float64),ks[:,None]),1);v=np.concatenate((o['past_v'].astype(np.float64),vs[:,None]),1)
        ek=k[np.arange(32)//4];ev=v[np.arange(32)//4]
        p=softmax(np.einsum('hd,hsd->hs',q,ek)/np.sqrt(128),axis=1)
        h=np.einsum('hs,hsd->hd',p,ev).reshape(-1);head.append(h)
        if keep:
            padded=np.zeros((32,maxn),np.float32);padded[:,:position+1]=p;probs.append(padded)
            gh=np.stack([np.einsum('hs,hsd->hd',p[:,o['groups']==g],ev[:,o['groups']==g]).reshape(-1) for g in range(6)])
            vectors.append(gh)
    output=np.stack(head)@wo.T
    extra={}
    if keep:
        gh=np.stack(vectors);extra={'probabilities':np.stack(probs),'source_vectors':(gh.reshape(-1,4096)@wo.T).reshape(len(indices),6,2560).astype(np.float32),'head_prediction':np.stack(head).astype(np.float32)}
        assert np.max(np.abs(extra['source_vectors'].astype(np.float64).sum(1)-output))<1e-4
    return output,extra


def fit_kernel(gram,tr,va,te):
    e,q=np.linalg.eigh((gram[np.ix_(tr,tr)]+gram[np.ix_(tr,tr)].T)/2)
    assert e.min()>-1e-7*max(1.,float(e.max()));e=np.maximum(e,0)
    return e,q,gram[np.ix_(va,tr)]@q,gram[np.ix_(te,tr)]@q


def df_ridge(e,df):
    assert np.count_nonzero(e>1e-10)>df
    lo,hi=-20.,20.
    for _ in range(90):
        mid=(lo+hi)/2;value=np.sum(e/(e+np.exp(mid)))
        if value>df:lo=mid
        else:hi=mid
    return float(np.exp((lo+hi)/2))


def metrics(target,prediction,tr,te,rows):
    summary,arrays=errors(target[te],prediction,np.mean(target[tr]**2,0));per=np.mean((target[te]-prediction)**2,1)
    for key in ('family','order','language','unit'):
        summary['by_'+key]={str(v):{'n':int(mask.sum()),'mse':float(per[mask].mean())} for v in sorted({r[key] for r in rows},key=str) if (mask:=np.array([rows[i][key]==v for i in te])).any()}
    return summary,arrays


def main():
    protocol=prepare();started=time.monotonic();rows,objects,data,oracle=collect();tr,va,te=splits(rows);assert len(te)>0 and len(tr)>64
    assert shutil.disk_usage(OUT).free>protocol['resources']['free_floor']
    wo=checkpoint('model.layers.23.self_attn.o_proj.weight').float().numpy().astype(np.float64)
    h=data['h12'].astype(np.float64);hscale=max(float(np.sqrt(np.mean(np.sum(h[tr]**2,1)))),1e-12);hg=h@h.T/hscale**2
    cg,cs=cache_gram(objects,tr);base={'H12':hg,'H12_fullpastKV':(hg+cg['past_k']+cg['past_v'])/3}
    npz(OUT/'input_grams.npz',H12=hg,past_K=cg['past_k'],past_V=cg['past_v'],train=tr,validation=va,test=te,H12_scale=np.array(hscale),K_scale=np.array(cs['past_k']),V_scale=np.array(cs['past_v']))
    norms=(np.linalg.norm(data['qkv'][tr,:4096].reshape(-1,32,128),axis=2).mean(0),np.linalg.norm(data['qkv'][tr,4096:5120].reshape(-1,8,128),axis=2).mean(0))
    target=data['attention'];reports=[];native=[]
    for i,(q,k,v) in enumerate(oracle):
        p=softmax(np.einsum('hd,hsd->hs',q,k[np.arange(32)//4])/np.sqrt(128),axis=1)
        head=np.einsum('hs,hsd->hd',p,v[np.arange(32)//4]).reshape(-1);native.append(head)
    native=np.stack(native)@wo.T;floor,farr=metrics(target,native[te],tr,te,rows)
    npz(OUT/'predictions/native_arithmetic_floor.npz',prediction=native[te].astype(np.float32),target=target[te].astype(np.float32),test=te,**farr)
    save(OUT/'native_arithmetic_audit.json',{'timestamp':stamp(),'scope':'FP64 all-source QK/softmax/V/WO using native future factors: arithmetic floor, NOT forecast','test_metrics':floor,
      'all_cases':len(rows),'full_output_mse':float(np.mean((native-target)**2)),'max_abs':float(np.max(np.abs(native-target)))})
    def record(mid,pred,extra,meta):
        report,arr=metrics(target,pred,tr,te,rows);reports.append(dict(model=mid,**meta,**report))
        npz(OUT/f'predictions/{mid}.npz',prediction=pred.astype(np.float32),target=target[te].astype(np.float32),test=te,**arr,**extra)
        print('ATTENTION_FORECAST',mid,report['mse'],flush=True)
        assert time.monotonic()-started<protocol['resources']['maximum_seconds']
    # Strong cache-conditional prototypes: the fitted current query need not be learned
    # per-example for cached history alone to produce substantial state variation.
    for mode in ('global','family_language_order'):
        grouping=np.zeros(len(rows),int) if mode=='global' else np.array([sorted({group_key(r) for r in rows}).index(group_key(r)) for r in rows])
        means={g:(data['qkv'][tr[grouping[tr]==g]] if np.any(grouping[tr]==g) else data['qkv'][tr]).mean(0) for g in set(grouping)}
        assert all(np.isfinite(m).all() for m in means.values())
        factors=np.stack([means[grouping[i]] for i in te]);p,extra=compose(factors,te,objects,wo,keep=True)
        record('mean_'+mode+'_factors',p,extra,{'algorithm':'training_group_mean_factors_with_actual_past_KV','native_future_inputs':False})
        means={g:(data['head'][tr[grouping[tr]==g]] if np.any(grouping[tr]==g) else data['head'][tr]).mean(0) for g in set(grouping)}
        p=np.stack([means[grouping[i]] for i in te])@wo.T
        record('mean_'+mode+'_head',p,{}, {'algorithm':'training_group_mean_head','native_future_inputs':False})
    for input_name,g in base.items():
      for kind in ('linear','quadratic'):
        gram=1+g
        if kind=='quadratic':gram=gram*gram
        e,q,vq,tq=fit_kernel(gram,tr,va,te)
        for route in ('factors','direct_head_equal6144','direct_output'):
            y=data['qkv'] if route=='factors' else np.concatenate((data['head'],data['qkv'][:,4096:]),1) if route=='direct_head_equal6144' else target
            qty=q.T@y[tr];grid=[]
            for ridge in RIDGES:
                pv=vq@(qty/(e[:,None]+ridge))
                av=compose(pv,va,objects,wo)[0] if route=='factors' else pv[:,:4096]@wo.T if route=='direct_head_equal6144' else pv
                grid.append(float(np.mean((av-target[va])**2)))
            best=min(range(len(RIDGES)),key=lambda i:(grid[i],-RIDGES[i]));choices=[('validation',RIDGES[best])]
            if input_name=='H12_fullpastKV' and route!='direct_output':choices.append(('fixed_df64',df_ridge(e,64.)))
            for selection,ridge in choices:
                spectral=qty/(e[:,None]+ridge);p=tq@spectral;mid=f'{input_name}_{kind}_{route}_{selection}'
                meta={'input':input_name,'kernel':kind,'route':route,'selection':selection,'ridge':ridge,'validation_attention_grid':dict(zip(map(str,RIDGES),grid)),
                  'effective_degrees_of_freedom':float(np.sum(e/(e+ridge))),'linear_output_coordinates':y.shape[1],'past_cache_is_available_before_current_step':True,
                  'past_cache_used_by_factor_composer':route=='factors','H12_name_denotes_fitted_encoder_not_entire_factor_algorithm':True,
                  'native_current_L23_factors_are_inputs':False}
                npz(OUT/f'models/{mid}.npz',alpha=(q@spectral).astype(np.float32),ridge=np.array(ridge),train=tr,test=te,query_norm=norms[0],key_norm=norms[1])
                if route=='factors':
                    pred,extra=compose(p,te,objects,wo,keep=True);extra['factor_prediction']=p.astype(np.float32);record(mid,pred,extra,meta)
                    if selection=='validation':
                        pred,extra=compose(p,te,objects,wo,norms=norms,keep=True);extra['factor_prediction']=p.astype(np.float32)
                        record(mid+'_headnorm',pred,extra,dict(meta,headnorm='Training mean per-head demodulated Q/K norm; not a test-selected option'))
                else:
                    pred=p[:,:4096]@wo.T if route=='direct_head_equal6144' else p;record(mid,pred,{},meta)
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());campaign=sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())
    assert total<protocol['resources']['maximum_new_bytes'] and campaign<protocol['resources']['campaign_ceiling'] and shutil.disk_usage(OUT).free>protocol['resources']['free_floor']
    summary={'phase':2708,'timestamp':stamp(),'states':len(rows),'train':len(tr),'validation':len(va),'test':len(te),'reports':reports,'native_arithmetic_floor':floor,
      'new_bytes':total,'campaign_bytes':campaign,'elapsed_seconds':time.monotonic()-started,'limits':protocol['limits'],
      'scope':'Conditional native attention prediction at M observed Result-field onsets. Does not predict whole-answer correctness or replace previous cached computation.'}
    save(OUT/'result.json',summary);announce('n_cached_attention',state='analysis_complete',completed=len(rows),total=len(rows),test=len(te));print('CACHED_ATTENTION_COMPLETE',len(reports),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args();prepare() if args.prepare else main()
