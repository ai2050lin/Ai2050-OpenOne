"""Full-coordinate prefix binding kernels, with explicit available-information boundaries."""
import hashlib
from scipy.linalg import eigh
from rdc_relation_common import *
from phase2715_rdc_prefix_relations import PrefixRelations
from rdc_prefix_estimators import errors

KINDS=('current','history_mean','content','pos_binding','relation_binding','relation_bilinear','source_permuted','full_quadratic')
RIDGES=(.00001,.001,.1,1.,10.)
MIXES=(0.,.1,.25,.5,1.)


def features_at(h,prefix_ids,language,parser):
    assert len(h)==len(prefix_ids)
    h=np.asarray(h,np.float32);x=h[-1];past=h[:-1];p,pos=parser.weights(prefix_ids,language)
    if len(past):
        mean=past.mean(0);similarity=(past@x)/np.maximum(np.linalg.norm(past,axis=1)*np.linalg.norm(x),1e-8)
        weight=np.exp(similarity-similarity.max());content=weight@past/weight.sum()
        posmsg=pos[:-1].T@past/(1+pos[:-1].sum(0))[:,None];rel=p[:,1:]
        message=rel.T@past/(1+rel.sum(0))[:,None]
        seed=int(hashlib.sha256(np.asarray(prefix_ids,dtype='<i8').tobytes()).hexdigest()[:16],16)
        perm=rel[np.random.default_rng(seed).permutation(len(rel))];permuted=perm.T@past/(1+perm.sum(0))[:,None]
    else:
        mean=np.zeros_like(x);content=mean;posmsg=np.zeros((17,len(x)),np.float32);message=np.zeros((28,len(x)),np.float32);permuted=message
    return {'current':x,'history_mean':mean,'content':content,'pos_binding':posmsg,'relation_binding':message,'source_permuted':permuted}


def build(material,fresh=False,layer=12):
    parser=PrefixRelations();data=default_data();meta=[];target=[];next_state=[];actual23=[];actual24=[]
    for i,r in enumerate(material):
        z=load_field(r,fresh);h=unbits(z[f'h{layer}']);h23=unbits(z['h23']);h36=unbits(z['h36']);h24=unbits(z['h24'])
        for j,p in enumerate(r['anchors']):
            f=features_at(h[:p+1],r['prompt_ids'][:p+1],r['language'],parser)
            for k,v in f.items():data[k].append(v)
            target.append(np.concatenate([h23[p],h36[3*j]]));next_state.append(h36[3*j+1]);actual23.append(h23[p]);actual24.append(h24[3*j])
            meta.append({k:r[k] for k in ('sample_id','source_group','language','genre','split')}|{'anchor':j,'position':p,'known_next_token_id':r['prompt_ids'][p+1]})
        if i%128==127:print('RELATION_FEATURES',layer,i+1,len(material),flush=True)
    return {k:np.stack(v).astype(np.float32) for k,v in data.items()},np.stack(target).astype(np.float32),meta,{'next_h36':np.stack(next_state),'h23':np.stack(actual23),'h24':np.stack(actual24)}


def default_data():return {k:[] for k in ('current','history_mean','content','pos_binding','relation_binding','source_permuted')}


class Bank:
    def __init__(self,data,train=None,scales=None):
        self.data=data;self.scales={k:max(float(np.mean(np.sum(v[train].astype(float).reshape(len(train),-1)**2,axis=1))),1e-15) for k,v in data.items()} if scales is None else scales
    def dots(self,other=None):
        other=self if other is None else other;return {k:np.asarray(v.reshape(len(v),-1),np.float64)@np.asarray(other.data[k].reshape(len(other.data[k]),-1),np.float64).T/self.scales[k] for k,v in self.data.items()}
    @staticmethod
    def gram(dots,kind,mix=0.):
        x=dots['current']
        if kind=='current':return 1+x
        if kind=='full_quadratic':return (1+x)**2
        r=dots['relation_binding'] if kind=='relation_bilinear' else dots[kind]
        if kind=='relation_bilinear':r=r+x*r
        return 1+(x+mix*r)/(1+mix)


def splits(meta):return [np.array([i for i,r in enumerate(meta) if r['split']==s],int) for s in ('train','validation','test')]


def scaling(y,train,blocks):
    mean=y[train].mean(0,dtype=np.float64);scale=np.empty(y.shape[1])
    for a,b in blocks:scale[a:b]=max(float(np.sqrt(np.mean((y[train,a:b]-mean[a:b])**2))),1e-8)
    return mean,scale,(y-mean)/scale


def select(dots,kind,y,train,val,blocks,fixed_df=None,fixed_mix=None):
    mean,scale,target=scaling(y,train,blocks);candidates=[];best=None
    mixes=(0.,) if kind in ('current','full_quadratic') else MIXES
    if fixed_mix is not None:mixes=(fixed_mix,)
    for mix in mixes:
        gram=Bank.gram(dots,kind,mix);tt=gram[np.ix_(train,train)];e,q=eigh((tt+tt.T)*.5,check_finite=False)
        assert e.min()>-1e-6*max(float(e.max()),1.);e=np.maximum(e,0);qty=q.T@target[train];vq=gram[np.ix_(val,train)]@q
        ridges=RIDGES
        if fixed_df is not None:
            lo,hi=-30.,30.
            for _ in range(70):
                mid=(lo+hi)/2
                if np.sum(e/(e+np.exp(mid)))>fixed_df:lo=mid
                else:hi=mid
            ridges=(float(np.exp((lo+hi)/2)),)
        for ridge in ridges:
            spectral=qty/(e[:,None]+ridge);p=vq@spectral;loss=float(np.mean((p-target[val])**2));df=float(np.sum(e/(e+ridge)))
            candidates.append({'mix':mix,'ridge':ridge,'validation_normalized_MSE':loss,'effective_df':df})
            if best is None or (loss,-ridge,mix)<(best['validation_normalized_MSE'],-best['ridge'],best['mix']):
                best={'mix':mix,'ridge':ridge,'validation_normalized_MSE':loss,'effective_df':df};alpha=(q@spectral).astype(np.float32)
    return {'alpha':alpha,'mean':mean.astype(np.float32),'target_scale':scale.astype(np.float32)},best,candidates


def predict(model,gram):return (gram@model['alpha']*model['target_scale']+model['mean']).astype(np.float32)


def load_model(folder):
    with np.load(folder/'model.npz') as z:return {k:z[k] for k in z.files}


def paired_gain(reference,candidate,meta):
    groups=sorted({r['source_group'] for r in meta});delta=np.mean((reference-candidate),axis=1)
    values=np.array([delta[[r['source_group']==g for r in meta]].mean() for g in groups]);rng=np.random.default_rng(2716)
    boot=[rng.choice(values,len(values),replace=True).mean() for _ in range(2000)]
    return {'source_units':len(groups),'baseline_minus_candidate_MSE':float(values.mean()),'source_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}


def fit_main():
    out=BASE/'rules';out.mkdir(parents=True,exist_ok=True)
    if not (out/'protocol.json').exists():save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'kinds':KINDS,'mixes':MIXES,'ridges':RIDGES,
      'target':'H23 and raw H36, each entire2560 coordinates; joint mean training-RMS-normalized validation error.',
      'features':'All known H12 prefix sources; learned token-piece POS/relation probabilities use IDs and language only. Query is current source, message sources strictly earlier.',
      'message':'m_r=sum_s probability(r,s|prefix)*H12_s /(1+sum_s probability(r,s|prefix)); every source and coordinate retained.',
      'normalization':'Each full feature block divided in dot product by its training mean squared Frobenius norm, not per-test rescaling.',
      'bilinear':'Kernel product evaluates every current coordinate i by every relation-bound source coordinate j; exact, no low-rank approximation.',
      'source_permutation':'Independently prefix-hashed reassignment of relation probabilities among source positions. Global role label permutation would not be a valid kernel control.',
      'selection':'One mixture and ridge per route on validation only. Includes zero-history. Matched effective-df controls evaluated separately; no test retuning.',
      'boundary':'Conditional on real lower-layer field, not an embedding-only language model or autonomous rollout.'})
    data,y,meta,extra=build(rows());train,val,test=splits(meta);save(out/'rows.json',meta);bank=Bank(data,train);save(out/'scales.json',bank.scales);dots=bank.dots();result={};predictions={}
    for kind in KINDS:
        folder=out/kind
        if (folder/'result.json').exists():
            result[kind]=read(folder/'result.json');model=load_model(folder)
        else:
            model,best,grid=select(dots,kind,y,train,val,[(0,2560),(2560,5120)]);npz(folder/'model.npz',**model)
            result[kind]={'kind':kind,**best,'validation_grid':grid,'test':{}}
        gram=Bank.gram(dots,kind,result[kind]['mix']);p=predict(model,gram[:,train]);predictions[kind]=p[test]
        for name,a,b in [('h23',0,2560),('h36',2560,5120)]:
            report,arrays=errors(y[test,a:b],p[test,a:b],y[train,a:b],[meta[i] for i in test]);result[kind]['test'][name]=report
            npz(folder/f'test_{name}_errors.npz',**arrays)
        npz(folder/'predictions.npz',validation=p[val],test=p[test],validation_indices=val,test_indices=test)
        save(folder/'result.json',result[kind]);print('RELATION_FIT',kind,'mix',result[kind]['mix'],'df',round(result[kind]['effective_df'],2),'H36',result[kind]['test']['h36']['relative_mse'],flush=True);guard(20*1024**2)
    matched={};df=result['current']['effective_df']
    for kind in ('relation_bilinear','full_quadratic'):
        model,best,grid=select(dots,kind,y,train,val,[(0,2560),(2560,5120)],fixed_df=df,fixed_mix=result[kind]['mix']);p=predict(model,Bank.gram(dots,kind,best['mix'])[np.ix_(test,train)])
        matched[kind]={**best,'test_H23_MSE':float(np.mean((p[:,:2560]-y[test,:2560])**2)),'test_H36_MSE':float(np.mean((p[:,2560:]-y[test,2560:])**2))};npz(out/'matched_df'/f'{kind}.npz',prediction=p)
    comparisons={k:{layer:paired_gain((predictions['current'][:,a:b]-y[test,a:b])**2,(p[:,a:b]-y[test,a:b])**2,[meta[i] for i in test]) for layer,a,b in [('h23',0,2560),('h36',2560,5120)]} for k,p in predictions.items() if k!='current'}
    winner=min(KINDS,key=lambda k:(result[k]['validation_normalized_MSE'],KINDS.index(k)))
    save(out/'result.json',{'timestamp':stamp(),'candidates':result,'matched_effective_df':matched,'paired_test_comparisons':comparisons,'validation_MSE_winner':winner,
      'train_anchors':len(train),'validation_anchors':len(val),'test_anchors':len(test),'feature_scaling_sha':sha(out/'scales.json'),'protocol_sha':sha(out/'protocol.json')})
    print('RELATION_MAIN_FIT_COMPLETE',winner,usage(),flush=True)


if __name__=='__main__':fit_main()
