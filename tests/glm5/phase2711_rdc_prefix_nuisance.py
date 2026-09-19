"""Nuisance-adjusted all-coordinate profiles and sentence-cluster uncertainty."""
from scipy.linalg import solve
from rdc_prefix_common import *
OUT=CAMPAIGN/'atlas'


def main():
    rows=read(OUT/'anchors.json');tr=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    te=np.array([i for i,r in enumerate(rows) if r['split']=='test']);genres=sorted({r['genre'] for r in rows})
    h=np.empty((len(rows),4,2560),np.float32);x=[]
    for i,r in enumerate(rows):
        with np.load(CAMPAIGN/f'qwen4/fields/{r["sample_id"]}.npz') as z:h[i]=unbits(z['h'][[0,12,24,36],r['anchor']*3])
        a,b=r['char_span'];token=r['text'][a:b];position=r['position']
        x.append([1,float(r['language']=='zh'),np.log1p(position),np.log1p(position)**2,np.log1p(b),len(token),
          float(bool(re.fullmatch(r'\W+',token))),float(bool(re.fullmatch(r'\d+',token)))]+[float(r['genre']==g) for g in genres])
    x=np.asarray(x,float);scale=np.sqrt(np.mean(x[tr]**2,0));scale=np.maximum(scale,1.);x/=scale
    coef=solve(x[tr].T@x[tr]+.1*np.eye(x.shape[1]),x[tr].T@h[tr].reshape(len(tr),-1),assume_a='pos')
    residual=h-(x@coef).reshape(h.shape)
    g=np.array([r['graph']['features'] for r in rows]);records=[];arrays={}
    groups=sorted({rows[i]['sample_id'] for i in te});groupindex={v:k for k,v in enumerate(groups)}
    assignment=np.array([groupindex[rows[i]['sample_id']] for i in te]);rng=np.random.default_rng(2711)
    weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=500)
    for j,name in enumerate(GRAPH_NAMES[:18]):
        result={'feature':name};profiles={}
        for mode,field in [('raw',h),('nuisance_residual',residual)]:
            d={}
            for split,ii in [('train',tr),('test',te)]:
                mask=g[ii,j]>0
                if mask.any() and (~mask).any():d[split]=field[ii[mask]].mean(0)-field[ii[~mask]].mean(0)
            if len(d)!=2:result[mode]={'estimable':False};continue
            a=d['train'];b=d['test'];cos=(a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-30)
            result[mode]={'estimable':True,'cosine_H0_H12_H24_H36':cos.tolist()}
            arrays[name+'_'+mode+'_train']=a.astype(np.float32);arrays[name+'_'+mode+'_test']=b.astype(np.float32)
            # Bootstrap source units, not individual coordinate entries or correlated anchors.
            present=g[te,j]>0;pos=np.zeros((len(groups),2560));neg=np.zeros_like(pos);pc=np.zeros(len(groups));nc=np.zeros(len(groups))
            for k,ix in enumerate(te):
                gi=assignment[k]
                if present[k]:pos[gi]+=field[ix,3];pc[gi]+=1
                else:neg[gi]+=field[ix,3];nc[gi]+=1
            pcount=weights@pc;ncount=weights@nc;valid=(pcount>0)&(ncount>0)
            pp=weights[valid]@pos/pcount[valid,None]-weights[valid]@neg/ncount[valid,None]
            c=pp@a[3]/np.maximum(np.linalg.norm(pp,axis=1)*np.linalg.norm(a[3]),1e-30)
            result[mode]['H36_fixed_train_reference_cluster_bootstrap_CI95']=np.quantile(c,[.025,.975]).tolist()
            result[mode]['valid_bootstrap_replicates']=int(valid.sum())
        records.append(result)
    npz(OUT/'nuisance_profiles.npz',**arrays,nuisance_coefficients=coef.astype(np.float32),nuisance_input_scales=scale.astype(np.float32))
    save(OUT/'nuisance_audit.json',{'passed':True,'timestamp':stamp(),'source_sha':sha(Path(__file__)),'records':records,
      'nuisance_features':['constant','language_zh','log_token_position','squared_log_token_position','log_char_endpoint','current_token_char_length','current_token_punctuation_only','current_token_numeric_only']+['genre_'+g for g in genres],
      'fit':'Training-only ridge0.1, four declared checkpoints, all2560 output coordinates. Same coefficients applied to test.',
      'uncertainty':'500 source-unit bootstrap replicates of test profiles, holding training reference fixed; not full training uncertainty or a population-random corpus.',
      'limitations':['Residualization controls these measured factors only; incomplete lexical identity, semantic label confounding and selection remain.','Language and genre are not crossed; regularization handles collinearity but does not identify their separate causal effects.',
        'Rare cue profiles remain unstable even if a numerical interval is available.']})
    print('NUISANCE_COMPLETE',[(r['feature'],r.get('raw',{}).get('cosine_H0_H12_H24_H36',[None]*4)[-1],r.get('nuisance_residual',{}).get('cosine_H0_H12_H24_H36',[None]*4)[-1]) for r in records],flush=True)


if __name__=='__main__':main()
