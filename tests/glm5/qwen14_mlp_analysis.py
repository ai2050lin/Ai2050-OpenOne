"""Frozen, all-unit conditional-response analysis; native units, not invented labels."""
from collections import Counter
import gc
import time
import numpy as np
import qwen14_mlp_atlas as C
from qwen14_mlp_provenance import save_result
from mlp_tail_finalize import words

C.save_json=save_result


def freeze():
    p=C.OUT/'analysis-plan.json'
    if p.exists():raise RuntimeError('Analysis specification already frozen')
    C.save_json(p, {'schema':'qwen14_analysis_plan.v1','kind':'real','run_id':C.RUN,
        'plan_sha256':C.file_sha(C.OUT/'plan.json'),'source_sha256':C.file_sha(__file__),
        'fit':'discovery documents only; do not select using confirm or holdout',
        'pole_count':4,'keyword_candidates':6,'context_window':128,
        'ridge_per_sample':.01,'prediction_fields':['G','V','A'],
        'context_predictor':'128 current-token hash buckets + 128 preceding-token frequency buckets + source/position/intercept',
        'lexical_baseline':'256 current-token hash buckets + the same source/position/intercept (equal 262-column budget)',
        'negative_control':'discovery target document order cyclically shifted by 137 documents; no row-wise leakage',
        'evaluation':'per-unit MSE in both document holdouts; report all units, including failures; candidate improvement requires both ratios <=0.9',
        'keyword_scope':'discovery positive vs negative activation contexts; repeated word associations are not semantic definitions',
        'adjacent':'discovery-max absolute Pearson parent per target; frozen slope predicts each holdout; same-index and seeded random-parent baselines',
        'statistical_status':'descriptive census and predictive screening, not multiple-testing-controlled semantic confirmation',
        'causal_status':'requires separate pre-intervention candidate and dose freeze; no causal edges from screening'})
    print('Frozen analysis before response inspection',flush=True)


def data(block,field):
    return np.load(C.OUT/f'B{block}-{field}.npy',mmap_mode='r').reshape(-1,C.WIDTH)


def features():
    from transformers import AutoTokenizer
    plan=C.read(C.OUT/'plan.json');docs=C.read(C.OUT/'documents.json')['docs']
    token=np.load(C.OUT/'tokens.npy');tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    rows=len(token)*len(C.POSITIONS);current=np.asarray(token[:,C.POSITIONS]).reshape(-1)
    common=np.zeros((rows,6),np.float32);common[:,0]=1
    sources=sorted({d['source'] for d in docs});common[:,1]=np.tile(np.asarray(C.POSITIONS)/1024,len(token))
    for i,d in enumerate(docs):common[i*8:(i+1)*8,2+sources.index(d['source'])]=1
    baseline=np.zeros((rows,262),np.float32);conditional=baseline.copy()
    baseline[np.arange(rows),current%256]=1
    conditional[np.arange(rows),current%128]=1
    windows=[];df=Counter()
    for di,ids in enumerate(token):
        for pi,pos in enumerate(C.POSITIONS):
            row=di*8+pi
            conditional[row,128:256]=np.bincount(ids[pos-128:pos]%128,minlength=128)/128
            ws=words(tok.decode(ids[pos-128:pos+1].tolist()));windows.append(sorted(ws))
            if di<512:df.update(ws)
    baseline[:,256:]=common;conditional[:,256:]=common
    np.savez(C.OUT/'text-features.npz',baseline=baseline,conditional=conditional,current_token=current)
    C.save_json(C.OUT/'context-terms.json',{'windows':windows,'discovery_frequency':dict(df),
        'sources':sources,'plan_sha256':C.file_sha(C.OUT/'analysis-plan.json')})
    return baseline,conditional


def profiles():
    assert (C.OUT/'collection.json').exists()
    from transformers import AutoTokenizer
    C.read(C.OUT/'analysis-plan.json')
    baseline,conditional=features();del baseline,conditional
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    tokens=np.load(C.OUT/'tokens.npy');ids=tokens[:,C.POSITIONS].reshape(-1)
    decoded={int(v):tok.decode([int(v)]) for v in np.unique(ids)}
    cues=np.array([[any('\u4e00'<=c<='\u9fff' for c in decoded[int(t)]),any(c.isdigit() for c in decoded[int(t)]),
                    any(c in '，。！？；：,.!?;:' for c in decoded[int(t)]),'\n' in decoded[int(t)],
                    any(c in '{}[]();=<>`' for c in decoded[int(t)])] for t in ids],np.float32)
    terms=C.read(C.OUT/'context-terms.json');windows=[set(x) for x in terms['windows']];df=terms['discovery_frequency']
    splits={s:(lo*8,hi*8) for s,(lo,hi) in C.SPLITS.items()};start=time.monotonic()
    for block in C.BLOCKS:
        maps={f:data(block,f) for f in ['G','V','A']}
        profile={f:{k:[] for k in ['mean','std','min','max']} for f in maps}
        for f in maps:
            for s in splits:profile[f][s]={'positive_indices':[],'negative_indices':[]}
        regimes=[];corr={s:[] for s in splits}
        for lo in range(0,C.WIDTH,128):
            hi=min(C.WIDTH,lo+128);g=C.unbits(maps['G'][:,lo:hi]);v=C.unbits(maps['V'][:,lo:hi])
            regimes.extend(np.stack([((g>=0)&(v>=0)).mean(0),((g>=0)&(v<0)).mean(0),((g<0)&(v>=0)).mean(0),((g<0)&(v<0)).mean(0)],1).tolist())
            for s,(a,b) in splits.items():
                cx=cues[a:b]-cues[a:b].mean(0);gy=g[a:b]-g[a:b].mean(0)
                den=np.sqrt((cx*cx).sum(0)[:,None]*(gy*gy).sum(0)[None,:])
                corr[s].extend(np.divide(cx.T@gy,den,out=np.zeros((5,hi-lo)),where=den>1e-12).T.tolist())
            del g,v
            for f in maps:
                x=C.unbits(maps[f][:,lo:hi]);p=profile[f]
                for k,v in [('mean',x.mean(0,dtype=np.float64)),('std',x.std(0,dtype=np.float64)),('min',x.min(0)),('max',x.max(0))]:p[k].extend(v.tolist())
                for s,(a,b) in splits.items():
                    z=x[a:b];col=np.arange(hi-lo)[None,:]
                    high=np.argpartition(z,-4,axis=0)[-4:];low=np.argpartition(z,4,axis=0)[:4]
                    high=np.take_along_axis(high,np.argsort(z[high,col],axis=0)[::-1],axis=0)
                    low=np.take_along_axis(low,np.argsort(z[low,col],axis=0),axis=0)
                    p[s]['positive_indices'].extend((high+a).T.tolist());p[s]['negative_indices'].extend((low+a).T.tolist())
                del x
            if hi%2048==0 or hi==C.WIDTH:C.log('profiles',(block-28)*C.WIDTH+hi,8*C.WIDTH,start,block=block)
        cards=[]
        for u in range(C.WIDTH):
            pos=Counter();neg=Counter();seen=set()
            for idx in profile['A']['discovery']['positive_indices'][u]:
                if idx//8 not in seen:pos.update(windows[idx]);seen.add(idx//8)
            seen2=set()
            for idx in profile['A']['discovery']['negative_indices'][u]:
                if idx//8 not in seen2:neg.update(windows[idx]);seen2.add(idx//8)
            def score(w):return (pos[w]-.5*neg[w])*np.log1p(4096/max(1,df.get(w,0)))
            best=sorted((w for w in pos if df.get(w,0)>=3 and score(w)>0),key=lambda w:(-score(w),w))[:6]
            negative=[w for w,n in neg.most_common() if w not in best and df.get(w,0)>=3][:6]
            repeat={}
            for s in ['confirm','holdout']:
                ix=profile['A'][s]['positive_indices'][u]
                repeat[s]={'hits':sum(bool(windows[i]&set(best)) for i in ix),'n':4,'distinct_documents':len({i//8 for i in ix})}
            cards.append({'unit':u,'positive_keywords':best,'negative_keywords':negative,'repeat':repeat,
                'status':'candidate_context_association','semantic_confirmed':False,
                'description':'乘积正端候选语境：'+' / '.join(best[:3]) if best else '未提取稳定语境词；可查看完整原生响应。'})
        C.save_json(C.OUT/f'B{block}-profile.json',{'schema':'qwen14_layer_profile.v1','kind':'real','run_id':C.RUN,
            'model_snapshot':C.read(C.OUT/'plan.json')['model_snapshot'],'block':block,'width':C.WIDTH,'fields':profile,
            'sign_regime_fractions':regimes,'cue_correlations':corr,'cue_names':['当前token汉字','数字字符','标点字符','换行','代码符号']})
        C.save_json(C.OUT/f'B{block}-cues.json',{'units':cards,'scope':'Discovery-derived lexical cues, not semantic names; failures retained.'})
        del maps,profile,cards,corr;gc.collect()
    print('All native units profiled',flush=True)


def predict():
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    assert (C.OUT/'analysis-plan.json').exists()
    features_np=np.load(C.OUT/'text-features.npz');start=time.monotonic()
    X={key:features_np[key] for key in ['baseline','conditional']};operators={}
    # The two fitted methods have the same 262 feature columns and regularizer.
    for key in X:
        xx=X[key][:4096].astype(np.float64)
        penalty=np.eye(xx.shape[1])*.01*len(xx);penalty[-6,-6]=0
        operators[key]=np.linalg.solve(xx.T@xx+penalty,xx.T).astype(np.float32)
    results=[]
    with C.GpuLease(),torch.inference_mode():
        tx={k:torch.from_numpy(v).cuda() for k,v in X.items()}
        top={k:torch.from_numpy(v).cuda() for k,v in operators.items()}
        for b in C.BLOCKS:
            stats={}
            for f in ['G','V','A']:
                raw=data(b,f)
                arrays={s:{k:[] for k in ['mean_mse','lexical_mse','conditional_mse','shuffled_mse','conditional_over_lexical']} for s in ['confirm','holdout']}
                coeff=np.lib.format.open_memmap(C.OUT/f'B{b}-{f}-context-coeff.npy',mode='w+',dtype=np.float32,shape=(262,C.WIDTH))
                for lo in range(0,C.WIDTH,256):
                    hi=min(C.WIDTH,lo+256);y=torch.from_numpy(C.unbits(raw[:,lo:hi])).cuda();train=y[:4096]
                    weights={k:top[k]@train for k in top}
                    shuffled=torch.roll(train.reshape(512,8,-1),137,0).reshape(4096,-1)
                    wrong=top['conditional']@shuffled
                    coeff[:,lo:hi]=weights['conditional'].cpu().numpy()
                    for s,(a,z) in {'confirm':(4096,6144),'holdout':(6144,8192)}.items():
                        truth=y[a:z];mse=lambda pred:((truth-pred)**2).mean(0)
                        base=mse(train.mean(0));lex=mse(tx['baseline'][a:z]@weights['baseline']);cond=mse(tx['conditional'][a:z]@weights['conditional']);neg=mse(tx['conditional'][a:z]@wrong)
                        for k,v in [('mean_mse',base),('lexical_mse',lex),('conditional_mse',cond),('shuffled_mse',neg),('conditional_over_lexical',cond/lex.clamp_min(1e-20))]:arrays[s][k].extend(v.cpu().numpy().astype(float).tolist())
                    del y,train,weights,shuffled,wrong
                coeff.flush();del coeff
                good=np.ones(C.WIDTH,bool)
                for s in arrays:
                    a=arrays[s];good&=np.array(a['conditional_over_lexical'])<=.9
                    good&=np.array(a['conditional_mse'])<np.array(a['mean_mse'])
                    good&=np.array(a['conditional_mse'])<np.array(a['shuffled_mse'])
                stats[f]={'splits':arrays,'candidate_predictable':good.tolist(),'count':int(good.sum())}
                C.log('conditional_prediction',(b-28)*3+['G','V','A'].index(f)+1,24,start,block=b,field=f)
            C.save_json(C.OUT/f'B{b}-prediction.json',{'schema':'qwen14_unit_prediction.v1','block':b,'fields':stats,
                'scope':'Same-budget text-feature prediction; no answer labels or target activations used as input. Descriptive screening, no semantic confirmation.'})
            results.append({'block':b,'counts':{f:stats[f]['count'] for f in stats}})
            del stats;gc.collect();torch.cuda.empty_cache()
    C.save_json(C.OUT/'prediction-summary.json',{'layers':results,'analysis_plan_sha256':C.file_sha(C.OUT/'analysis-plan.json')})


def neighbors():
    """All pairs searched; keep frozen best-parent predictors, serve full rows on demand."""
    import torch
    start=time.monotonic();report=[]
    with C.GpuLease(),torch.inference_mode():
        for b in C.BLOCKS[:-1]:
            xraw=data(b,'A');yraw=data(b+1,'A')
            x=torch.from_numpy(C.unbits(xraw[:4096])).cuda();y=torch.from_numpy(C.unbits(yraw[:4096])).cuda()
            xm=x.mean(0);ym=y.mean(0);x-=xm;y-=ym
            xs=x.square().mean(0).sqrt();ys=y.square().mean(0).sqrt()
            normx=x/xs.clamp_min(1e-12);normy=y/ys.clamp_min(1e-12)
            best=torch.zeros(C.WIDTH,device='cuda');parent=torch.zeros(C.WIDTH,dtype=torch.long,device='cuda')
            for lo in range(0,C.WIDTH,256):
                corr=normx[:,lo:lo+256].T@normy/4096
                values,ix=corr.abs().max(0);signed=corr[ix,torch.arange(C.WIDTH,device='cuda')]
                choose=values>best.abs();best[choose]=signed[choose];parent[choose]=ix[choose]+lo
            parents=parent.cpu().numpy();chosen=best.cpu().numpy()
            slope=best*ys/xs[parent].clamp_min(1e-12);intercept=ym-slope*xm[parent]
            control_parents={'same_index':torch.arange(C.WIDTH,device='cuda'),
                'random_parent':torch.from_numpy(np.random.default_rng(20261109+b).integers(0,C.WIDTH,C.WIDTH)).cuda()}
            control_coeff={}
            for name,ix in control_parents.items():
                beta=(x[:,ix]*y).mean(0)/xs[ix].square().clamp_min(1e-20)
                control_coeff[name]=(beta,ym-beta*xm[ix])
            slopes=slope.cpu().numpy();intercepts=intercept.cpu().numpy()
            np.savez(C.OUT/f'B{b}-B{b+1}-predictor.npz',parent=parents,slope=slopes,intercept=intercepts,discovery_correlation=chosen)
            results={}
            for split,(a,z) in {'confirm':(4096,6144),'holdout':(6144,8192)}.items():
                xx=torch.from_numpy(C.unbits(xraw[a:z])).cuda();yy=torch.from_numpy(C.unbits(yraw[a:z])).cuda()
                pred=xx[:,parent]*slope+intercept
                mse=((pred-yy)**2).mean(0);baseline=((yy-ym)**2).mean(0)
                xp=xx[:,parent]-xx[:,parent].mean(0);yp=yy-yy.mean(0)
                rho=(xp*yp).mean(0)/(xp.square().mean(0)*yp.square().mean(0)).sqrt().clamp_min(1e-20)
                results[split]={'mse':mse.cpu().numpy().astype(float).tolist(),'mean_mse':baseline.cpu().numpy().astype(float).tolist(),
                    'correlation':rho.cpu().numpy().astype(float).tolist(),'ratio':(mse/baseline.clamp_min(1e-20)).cpu().numpy().astype(float).tolist()}
                for name,ix in control_parents.items():
                    beta,alpha=control_coeff[name]
                    err=((xx[:,ix]*beta+alpha-yy)**2).mean(0)
                    results[split][name+'_mse']=err.cpu().numpy().astype(float).tolist()
                del xx,yy,pred,xp,yp
            C.save_json(C.OUT/f'B{b}-B{b+1}-prediction.json',{'source':b,'target':b+1,'parents':parents.tolist(),
                'discovery_correlation':chosen.astype(float).tolist(),'splits':results,'causal_confirmed':False})
            report.append({'source':b,'target':b+1,'holdout_mean_ratio':float(np.mean(results['holdout']['ratio'])),
                'holdout_better_than_mean':int(np.sum(np.asarray(results['holdout']['ratio'])<.9))})
            del x,y,xm,ym,xs,ys,normx,normy,best,parent,results;gc.collect();torch.cuda.empty_cache()
            C.log('neighbor_predictors',b-27,7,start)
    # Free rollout starts with true B28 A, then uses only its own predicted states.
    rollout=C.unbits(data(28,'A')[6144:]);rows=[]
    for b in C.BLOCKS[:-1]:
        p=np.load(C.OUT/f'B{b}-B{b+1}-predictor.npz');rollout=rollout[:,p['parent']]*p['slope']+p['intercept']
        truth=C.unbits(data(b+1,'A')[6144:]);mean=C.unbits(data(b+1,'A')[:4096]).mean(0)
        ratio=((rollout-truth)**2).mean(0)/np.maximum(((truth-mean)**2).mean(0),1e-20)
        rows.append({'target':b+1,'mean_mse_ratio':float(ratio.mean()),'unit_ratios':ratio.astype(float).tolist()})
    C.save_json(C.OUT/'neighbor-summary.json',{'one_step':report,'rollout':rows,
        'scope':'Prediction from observed B28 A. Not text-to-output prediction, not a causal graph; unstable rollout retained.'})


def readout():
    import torch
    start=time.monotonic()
    with C.GpuLease(),torch.inference_mode():
        wu=C.weight('lm_head.weight').cuda().float();gamma=C.weight('model.norm.weight').cuda().float()
        for b in C.BLOCKS:
            down=torch.from_numpy(C.unbits(np.load(C.OUT/f'B{b}-W-down.npy'))).cuda()
            ids_high=[];ids_low=[];values_high=[];values_low=[]
            for lo in range(0,C.WIDTH,128):
                r=wu@(gamma[:,None]*down[:,lo:lo+128])
                positive,pi=r.topk(5,dim=0);negative,ni=(-r).topk(5,dim=0)
                ids_high.extend(pi.T.cpu().tolist());ids_low.extend(ni.T.cpu().tolist())
                values_high.extend(positive.T.cpu().tolist());values_low.extend((-negative).T.cpu().tolist())
                if lo%4096==0:C.log('parameter_readout',(b-28)*C.WIDTH+lo+128,8*C.WIDTH,start)
            C.save_json(C.OUT/f'B{b}-readout.json',{'positive_ids':ids_high,'negative_ids':ids_low,
                'positive_values':values_high,'negative_values':values_low,
                'scope':'W_unembed @ (final_gamma * W_down[:,u]); direct-readout proxy omits intervening layers, context RMS denominator and attention. Not a semantic label or actual next-token prediction.'})
            del down;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    import sys
    {'freeze':freeze,'profiles':profiles,'predict':predict,'neighbors':neighbors,'readout':readout}[sys.argv[1]]()
