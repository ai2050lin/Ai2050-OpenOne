"""Reanalyse exposed old fields: diagnose state sufficiency, never new confirmation."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
import branch_reuse14 as B
import recursive_branch14 as R
import recursive_branch14_analysis as A


def run():
    B.configure(300*2**20);start=time.monotonic()
    selection=B.read(B.PREVIOUS/'selection.json')
    assert R.OUT==B.PREVIOUS
    for path,sha in selection['model_hashes'].items():assert B.file_sha(B.PREVIOUS/path)==sha
    old=B.read(B.PREVIOUS/'response-report.json');rows,meta=A.load('discovery')
    assert meta['model_snapshot']==B.SNAPSHOT
    results=[];diagnostics=[]
    with threadpool_limits(limits=4):
      for h in R.CANDIDATES:
        ix=np.array([i for i,r in enumerate(rows) if r['hypothesis']==h]);chain=selection['backward'][h]
        root=chain['roots'];target=A.target('discovery',39,root)[ix]
        hidden=np.load(B.PREVIOUS/'discovery/hidden-all.npy',mmap_mode='r')
        predictors={}
        for boundary in [36,37,38,39]:
            x=B.unbits(hidden[ix,boundary,0]).astype(np.float64)
            predictors[f'residual_{boundary}_to_B39']=A.regression(x,target)
        pooled=np.load(B.PREVIOUS/'discovery/prefix-embedding.npy',mmap_mode='r')
        predictors['pooled_embedding']=A.regression(np.asarray(pooled[ix],np.float64),target)
        np.savez_compressed(B.OUT/f'development-{h}-predictors.npz',**{
            name+'__'+key:value for name,p in predictors.items() for key,value in p.items()})
        for split in R.SPLITS[1:]:
            sr,sm=A.load(split);si=np.array([i for i,r in enumerate(sr) if r['hypothesis']==h])
            yy=A.target(split,39,root)[si]
            first=chain['edges'][-1]
            original=A.field(split,36,'A')[si][:,first['parents']]
            for mode in ['direct_A','composed_A']:
                current=original
                for edge in reversed(chain['edges']):
                    model=A.get_model(edge['fit_prefix']+'-parent');prediction=A.predict(model,current)
                    if mode=='composed_A':prediction=A.compose(prediction)
                    current=prediction.reshape(len(si),-1,3)[:,:,2]
                metrics=A.error(yy,prediction,model)
                results.append(dict(hypothesis=h,split=split,method='free_chain_'+mode,metrics=metrics))
                if mode=='composed_A':
                    prior=next(x for x in old['backward_chains'] if x['hypothesis']==h and x['split']==split)
                    assert abs(metrics['standardized_MSE']-prior['free_rollout']['standardized_MSE'])<1e-6
            hs=np.load(B.PREVIOUS/split/'hidden-all.npy',mmap_mode='r')
            for boundary in [36,37,38,39]:
                name=f'residual_{boundary}_to_B39';model=predictors[name]
                prediction=A.predict(model,B.unbits(hs[si,boundary,0]).astype(np.float64))
                results.append(dict(hypothesis=h,split=split,method=name,metrics=A.error(yy,prediction,model)))
            emb=np.load(B.PREVIOUS/split/'prefix-embedding.npy',mmap_mode='r')
            model=predictors['pooled_embedding'];prediction=A.predict(model,np.asarray(emb[si],np.float64))
            results.append(dict(hypothesis=h,split=split,method='pooled_embedding',metrics=A.error(yy,prediction,model)))
            B.log('exposed_field_state_diagnostics',len(results),42,start,hypothesis=h,split=split)
    B.save('development-state-diagnostics.json',dict(kind='real',status='EXPOSED_OLD_MATERIAL_DEVELOPMENT_ONLY',
        results=results,source_sha256=B.file_sha(__file__),
        old_inputs={p:B.file_sha(B.PREVIOUS/p) for p in ['selection.json','response-report.json','materials.json','native-manifest.json']},
        scope='Original frozen roots and fits retained. Direct-A rollout diagnoses an algorithm choice. Residual-state regressions are high-dimensional strong context baselines, not neuron graphs or causal mechanisms.',
        selection_warning='All old holdouts were exposed in Phase 2923 and are development data for every new hypothesis. No new confirmatory pass or statistical promotion.'))


if __name__=='__main__':run()
