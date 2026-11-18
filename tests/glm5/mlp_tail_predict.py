"""Prospectively specified sparse native wiring, baselines and free three-layer rollout."""
import argparse
import gc
import math
import time
import numpy as np
import mlp_tail_goodfire as C


def freeze():
    path=C.OUT/'prediction-plan.json'
    assert not path.exists()
    C.save_json(path,{'schema':'mlp_tail_prediction_plan.v1','kind':'real','run_id':C.RUN,
       'model_snapshot':C.SNAPSHOT,'source_sha256':C.file_sha(__file__),'seed':C.SEED,
       'parents':8,'selection':'Largest absolute discovery A-to-A Pearson coefficients per target; all full matrices retained.',
       'fit':'Per target ridge with discovery-normalized source A and lambda=.01*n; fit G,V,A separately, same selected parents.',
       'baselines':['discovery mean','same-index affine predictor','eight seeded random parents','token means (>=8 discovery occurrences) with language/position fallback'],
       'primary':'Paired document-bootstrap FVU improvement over token baseline, confirm/holdout separate; 98.3333% CIs for 3 adjacent pairs.',
       'exposure':'Existing bank held out from this estimator only. Not a new independent semantic confirmation.',
       'rollout':'Observed B32 A only, then recursively predicted A33--A35; direct linear and composed SiLU(Ghat)*Vhat compared.',
       'semantic_or_causal_promotion':False,'frozen_before_association_selection':not (C.OUT/'associations.json').exists()})


def normalized_tensor(x):
    import torch
    y=torch.from_numpy(x).cuda();mean=y.mean(0);std=y.std(0,unbiased=False).clamp_min(1e-5)
    return (y-mean)/std,mean,std


def fit_targets(x,y,parents):
    """Independent batched ridge regressions; target columns never input the fit."""
    import torch
    n,d=y.shape;k=parents.shape[1];coef=torch.empty((d,k),device='cuda');mu=y.mean(0)
    for lo in range(0,d,128):
        hi=min(d,lo+128);p=parents[lo:hi]
        design=x[:,p].permute(1,2,0).contiguous() # target,k,sample
        gram=design@design.transpose(1,2)/n+.01*torch.eye(k,device='cuda')[None]
        rhs=(design@(y[:,lo:hi]-mu[lo:hi]).T[:,:,None])/n
        coef[lo:hi]=torch.linalg.solve(gram,rhs).squeeze(2)
    return coef,mu


def predict(x,parents,coef,mu):
    import torch
    out=torch.empty((len(x),len(parents)),device='cuda')
    for lo in range(0,len(parents),128):
        hi=min(len(parents),lo+128)
        out[:,lo:hi]=(x[:,parents[lo:hi]]*coef[None,lo:hi,:]).sum(2)+mu[None,lo:hi]
    return out


def score(pred,truth,mean):
    import torch
    err=(pred-truth).square();den=(truth-mean).square()
    unit=err.sum(0)/den.sum(0).clamp_min(1e-20)
    doc=err.reshape(-1,16,C.WIDTH).sum((1,2)).double().cpu().numpy()
    base=den.reshape(-1,16,C.WIDTH).sum((1,2)).double().cpu().numpy()
    return {'fvu':float(err.sum()/den.sum().clamp_min(1e-20)),'unit_fvu':unit.cpu().numpy(),'doc_error':doc,'doc_baseline':base}


def main():
    import torch
    plan=C.read(C.OUT/'prediction-plan.json');assert plan['source_sha256']==C.file_sha(__file__)
    assert C.read(C.OUT/'associations.json')['kind']=='real'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic();rng=np.random.default_rng(C.SEED);reports=[]
    ids=np.load(C.OUT/'token_ids.npy');langs=np.repeat([int('zh' in d['source']) for d in C.read(C.BANK/'source-documents.json')['docs']],16)
    keys,counts=np.unique(ids[:16384],return_counts=True);keys=keys[counts>=8]
    with C.GpuLease(),torch.inference_mode():
        for si,(source,target) in enumerate(zip(C.BLOCKS,C.BLOCKS[1:])):
            corr=torch.from_numpy(np.asarray(np.load(C.OUT/f'B{source}-B{target}-rho-discovery.npy',mmap_mode='r')).copy()).cuda()
            parents=corr.abs().topk(8,dim=0).indices.T.contiguous();del corr
            random_parents=torch.tensor(np.stack([rng.choice(C.WIDTH,8,replace=False) for _ in range(C.WIDTH)]),device='cuda')
            x,xmean,xstd=normalized_tensor(C.data(source,'A','discovery'))
            params={'parents':parents.cpu().numpy(),'random_parents':random_parents.cpu().numpy(),'xmean':xmean.cpu().numpy(),'xstd':xstd.cpu().numpy()}
            for field in ['G','V','A']:
                y=torch.from_numpy(C.data(target,field,'discovery')).cuda()
                coef,mu=fit_targets(x,y,parents);params[field+'_coef']=coef.cpu().numpy();params[field+'_mean']=mu.cpu().numpy()
                if field=='A':
                    random_coef,_=fit_targets(x,y,random_parents)
                    params['random_coef']=random_coef.cpu().numpy()
                    params['same_coef']=((x*y).mean(0)).cpu().numpy()
                    ycpu=y.cpu().numpy();means=[]
                    for token in keys:means.append(ycpu[ids[:16384]==token].mean(0))
                    params['token_keys']=keys;params['token_means']=np.asarray(means)
                    params['language_means']=np.stack([ycpu[langs[:16384]==l].mean(0) for l in [0,1]])
                    params['position_means']=ycpu.reshape(-1,16,C.WIDTH).mean(0)
                    del ycpu,means
                del y,coef,mu
                C.log('fit_sparse_native',si*3+['G','V','A'].index(field)+1,9,started,source=source,target=target)
            np.savez(C.OUT/f'B{source}-B{target}-predictor.npz',**params)
            del x
            for split,a,b in [('confirm',16384,24576),('holdout',24576,32768)]:
                x=(torch.from_numpy(C.data(source,'A',split)).cuda()-xmean)/xstd
                target_a=torch.from_numpy(C.data(target,'A',split)).cuda();mean=torch.from_numpy(params['A_mean']).cuda()
                predictions={}
                for field in ['G','V','A']:
                    predictions[field]=predict(x,parents,torch.from_numpy(params[field+'_coef']).cuda(),torch.from_numpy(params[field+'_mean']).cuda())
                position_offset=np.tile(params['position_means']-params['A_mean'],((b-a)//16,1))
                token_pred=params['language_means'][langs[a:b]].copy();lookup={int(k):i for i,k in enumerate(keys)}
                for row,token in enumerate(ids[a:b]):
                    if int(token) in lookup:token_pred[row]=params['token_means'][lookup[int(token)]]
                token_pred+=position_offset
                methods={'sparse_direct':predictions['A'],'sparse_composed':torch.nn.functional.silu(predictions['G'])*predictions['V'],
                    'same_index':x*torch.from_numpy(params['same_coef']).cuda()+mean,
                    'random8':predict(x,random_parents,torch.from_numpy(params['random_coef']).cuda(),mean),
                    'token':torch.from_numpy(token_pred).cuda(),'mean':mean.expand_as(target_a)}
                results={k:score(v,target_a,mean) for k,v in methods.items()}
                saving={}
                for name,value in results.items():
                    saving[name+'_unit_fvu']=value['unit_fvu'];saving[name+'_doc_error']=value['doc_error']
                saving['doc_baseline']=results['token']['doc_baseline']
                np.savez(C.OUT/f'B{source}-B{target}-{split}-scores.npz',**saving)
                ratio=[]
                for _ in range(1500):
                    ix=rng.integers(0,len(results['token']['doc_error']),len(results['token']['doc_error']))
                    den=results['token']['doc_baseline'][ix].sum()
                    ratio.append((results['token']['doc_error'][ix].sum()-results['sparse_direct']['doc_error'][ix].sum())/den)
                ci=np.quantile(ratio,[.05/6,1-.05/6]).tolist()
                reports.append({'source':source,'target':target,'split':split,'n_docs':(b-a)//16,
                    'fvu':{k:v['fvu'] for k,v in results.items()},'direct_gain_over_token_ci98333':ci,
                    'units_direct_fvu_below_one':int((results['sparse_direct']['unit_fvu']<1).sum()),
                    'largest_error_doc_indices':(np.argsort(results['sparse_direct']['doc_error'])[-16:][::-1]+a//16).tolist()})
                C.log('heldout_one_step',len(reports),6,started,source=source,target=target,split=split,fvu=reports[-1]['fvu'])
                del x,target_a,predictions,methods,results,token_pred;torch.cuda.empty_cache()
            del params,parents,random_parents,xmean,xstd;gc.collect();torch.cuda.empty_cache()
        roll=[]
        for split in ['confirm','holdout']:
            initial=torch.from_numpy(C.data(32,'A',split)).cuda()
            direct=initial.clone();composed=initial.clone();del initial
            for source,target in zip(C.BLOCKS,C.BLOCKS[1:]):
                with np.load(C.OUT/f'B{source}-B{target}-predictor.npz') as z:
                    params={k:torch.from_numpy(z[k]).cuda() for k in ['parents','xmean','xstd','A_coef','A_mean','G_coef','G_mean','V_coef','V_mean']}
                xd=(direct-params['xmean'])/params['xstd'];xc=(composed-params['xmean'])/params['xstd']
                direct=predict(xd,params['parents'],params['A_coef'],params['A_mean'])
                g=predict(xc,params['parents'],params['G_coef'],params['G_mean']);v=predict(xc,params['parents'],params['V_coef'],params['V_mean'])
                composed=torch.nn.functional.silu(g)*v
                truth=torch.from_numpy(C.data(target,'A',split)).cuda()
                metrics={k:score(p,truth,params['A_mean'])['fvu'] for k,p in [('direct',direct),('composed',composed)]}
                roll.append({'source_initial':32,'target':target,'steps':target-32,'split':split,'fvu':metrics})
                C.log('free_internal_rollout',len(roll),6,started,**roll[-1])
                del params,xd,xc,g,v,truth;torch.cuda.empty_cache()
            del direct,composed;torch.cuda.empty_cache()
    C.save_json(C.OUT/'prediction.json',{'schema':'mlp_tail_prediction.v1','kind':'real','run_id':C.RUN,'model_snapshot':C.SNAPSHOT,
             'one_step':reports,'rollout':roll,'seconds':time.monotonic()-started,
             'scope':'Conditional internal-state prediction, initialized from observed source A. No independent next-token, semantic or causal closure claimed.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.stage=='freeze' else main()
