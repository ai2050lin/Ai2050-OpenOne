"""Frozen whole-coordinate analyses for the B18--B28 native study."""
import argparse
import gc
import time
import numpy as np
import goodfire_middle as M
C = M.C


def data(block, field):
    return np.load(M.OUT/f'B{block}-{field}.npy',mmap_mode='r').reshape(-1,C.WIDTH)


def standardize(x):
    x=x.astype(np.float64).copy();mean=x[:4096].mean(0);std=x[:4096].std(0)
    keep=std>1e-12;mean[256]=0;std[256]=1;keep[256]=True
    x[:,keep]=(x[:,keep]-mean[keep])/std[keep];x[:,~keep]=0
    return x.astype(np.float32),dict(mean=mean.tolist(),std=std.tolist(),nonconstant=keep.tolist())


def operator(x, intercept):
    train=x[:4096].astype(np.float64)
    penalty=np.eye(train.shape[1])*.01*4096;penalty[intercept,intercept]=0
    return np.linalg.solve(train.T@train+penalty,train.T).astype(np.float32)


def profiles():
    import qwen14_mlp_analysis as A
    C.save_json=M.write;A.profiles()


def regimes():
    # Reuse the immutable arithmetic implementation with explicit runtime routing.
    import qwen14_goodfire_analysis as R
    R.E.OUT=M.OUT;R.E.ALL=M.BLOCKS;R.E.data=data;R.E.save=M.write
    def progress(stage,done=0,total=0,start=None,**kw):
        M.log(stage,done+10,11,start,**kw)
    R.E.log=progress;R.regimes()


def predict():
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4)
    features=np.load(M.OUT/'text-features.npz')
    rawx={k:features[k] for k in ['baseline','conditional']}
    rawx['matched_current']=rawx['conditional'].copy();rawx['matched_current'][:,128:256]=0
    X={};scales={}
    for k,x in rawx.items():X[k],scales[k]=standardize(x)
    ops={k:operator(x,256) for k,x in X.items()}
    background=rawx['baseline'][:,256:].copy();background_op=operator(background,0)
    current=features['current_token'];unique,inverse=np.unique(current,return_inverse=True)
    count=np.bincount(inverse[:4096],minlength=len(unique)).astype(np.float32)
    M.write(M.OUT/'prediction-feature-scales.json',dict(scales=scales,unique_current_tokens=len(unique),
        holdout_unseen_token_rows=int((count[inverse[6144:]]==0).sum()),
        token_shrinkage=10.,fit_rows=[0,4096]))
    summary=[];start=time.monotonic()
    with C.GpuLease(),torch.inference_mode():
        tx={k:torch.from_numpy(x).cuda() for k,x in X.items()}
        op={k:torch.from_numpy(x).cuda() for k,x in ops.items()}
        bg=torch.from_numpy(background).cuda();bgop=torch.from_numpy(background_op).cuda()
        token_index=torch.from_numpy(inverse).cuda();counts=torch.from_numpy(count).cuda()
        for b in M.BLOCKS:
            raw={f:data(b,f) for f in ['G','V','A']};arrays={}
            names=['mean','baseline','conditional','matched_current','shuffled','exact_token_background']
            for f in raw:
                for s in ['confirm','holdout']:
                    for k in names:arrays[f+'_'+s+'_'+k+'_mse']=np.empty(C.WIDTH,np.float32)
            for s in ['confirm','holdout']:
                arrays['A_'+s+'_composed_mse']=np.empty(C.WIDTH,np.float32)
            coefficients={f:np.lib.format.open_memmap(M.OUT/f'B{b}-{f}-context-coeff.npy',
                mode='w+',dtype=np.float32,shape=(262,C.WIDTH)) for f in raw}
            for lo in range(0,C.WIDTH,256):
                hi=min(C.WIDTH,lo+256);responses={};conditional_predictions={}
                for f in raw:
                    y=torch.from_numpy(C.unbits(raw[f][:,lo:hi])).cuda();train=y[:4096]
                    weights={k:o@train for k,o in op.items()}
                    coefficients[f][:,lo:hi]=weights['conditional'].cpu().numpy()
                    wrong=op['conditional']@torch.roll(train.reshape(512,8,-1),137,0).reshape(4096,-1)
                    bgweight=bgop@train;residual=train-bg[:4096]@bgweight
                    token_sum=torch.zeros((len(unique),hi-lo),device='cuda')
                    token_sum.index_add_(0,token_index[:4096],residual)
                    token_mean=token_sum/(counts[:,None]+10.)
                    predictions={k:tx[k]@weights[k] for k in weights}
                    predictions['exact_token_background']=bg@bgweight+token_mean[token_index]
                    predictions['shuffled']=tx['conditional']@wrong
                    for s,(a,z) in {'confirm':(4096,6144),'holdout':(6144,8192)}.items():
                        truth=y[a:z]
                        arrays[f+'_'+s+'_mean_mse'][lo:hi]=((truth-train.mean(0))**2).mean(0).cpu().numpy()
                        for k,pred in predictions.items():
                            arrays[f+'_'+s+'_'+k+'_mse'][lo:hi]=((truth-pred[a:z])**2).mean(0).cpu().numpy()
                    conditional_predictions[f]=predictions['conditional'][4096:].clone()
                    if f=='A':responses['A']=y[4096:].clone()
                    del y,train,weights,wrong,token_mean,token_sum,predictions,residual
                composed=torch.nn.functional.silu(conditional_predictions['G'])*conditional_predictions['V']
                for s,(a,z) in {'confirm':(0,2048),'holdout':(2048,4096)}.items():
                    arrays['A_'+s+'_composed_mse'][lo:hi]=((responses['A'][a:z]-composed[a:z])**2).mean(0).cpu().numpy()
                del responses,conditional_predictions,composed
                if lo%2048==0:M.log('response_prediction',(b-18)*C.WIDTH+hi,11*C.WIDTH,start,block=b)
            for coeff in coefficients.values():coeff.flush()
            del coefficients
            counts_good={};layer=dict(block=b,counts={},holdout={})
            for f in raw:
                good=np.ones(C.WIDTH,bool)
                for s in ['confirm','holdout']:
                    cond=arrays[f+'_'+s+'_conditional_mse'];lex=arrays[f+'_'+s+'_baseline_mse']
                    good&=cond<=.9*lex
                    for k in ['mean','matched_current','shuffled','exact_token_background']:
                        good&=cond<arrays[f+'_'+s+'_'+k+'_mse']
                arrays[f+'_candidate']=good;counts_good[f]=int(good.sum())
                mean=arrays[f+'_holdout_mean_mse'];cond=arrays[f+'_holdout_conditional_mse']
                layer['holdout'][f]=dict(per_unit_mean_ratio=float(np.mean(cond/np.maximum(mean,1e-20))),
                    total_MSE_ratio=float(cond.sum()/max(mean.sum(),1e-20)))
            composed_good=np.ones(C.WIDTH,bool)
            for s in ['confirm','holdout']:
                composed_good&=arrays['A_'+s+'_composed_mse']<=.9*arrays['A_'+s+'_conditional_mse']
                composed_good&=arrays['A_'+s+'_composed_mse']<arrays['A_'+s+'_mean_mse']
            arrays['A_composed_candidate']=composed_good
            layer.update(counts=counts_good,composed_candidates=int(composed_good.sum()))
            np.savez_compressed(M.OUT/f'B{b}-prediction.npz',**arrays)
            summary.append(layer);del raw,arrays;gc.collect();torch.cuda.empty_cache()
            M.log('predicted_layers',b-17,11,start)
    M.write(M.OUT/'prediction-summary.json',dict(layers=summary,
        scope='Frozen discovery-only scaling; exact-token plus background control; heldout documents previously exposed elsewhere. No semantic confirmation.',
        composition='SiLU(predicted G)*predicted V; not equal to the conditional expectation of A in general.'))


def neighbors():
    import torch
    start=time.monotonic();report=[];torch.set_num_threads(4)
    with C.GpuLease(),torch.inference_mode():
        for b in M.BLOCKS[:-1]:
            xraw=data(b,'A');yraw=data(b+1,'A')
            x=torch.from_numpy(C.unbits(xraw[:4096])).cuda();y=torch.from_numpy(C.unbits(yraw[:4096])).cuda()
            xm=x.mean(0);ym=y.mean(0);x-=xm;y-=ym
            xs=x.square().mean(0).sqrt();ys=y.square().mean(0).sqrt()
            nx=x/xs.clamp_min(1e-12);ny=y/ys.clamp_min(1e-12)
            best=torch.zeros(C.WIDTH,device='cuda');parent=torch.zeros(C.WIDTH,dtype=torch.long,device='cuda')
            for lo in range(0,C.WIDTH,256):
                corr=nx[:,lo:lo+256].T@ny/4096
                val,ix=corr.abs().max(0);signed=corr[ix,torch.arange(C.WIDTH,device='cuda')]
                choose=val>best.abs();best[choose]=signed[choose];parent[choose]=ix[choose]+lo
            slope=best*ys/xs[parent].clamp_min(1e-12);intercept=ym-slope*xm[parent]
            controls={'same_index':torch.arange(C.WIDTH,device='cuda'),
                'random_parent':torch.from_numpy(np.random.default_rng(M.SEED+b).integers(0,C.WIDTH,C.WIDTH)).cuda()}
            coeff={}
            for k,ix in controls.items():
                beta=(x[:,ix]*y).mean(0)/xs[ix].square().clamp_min(1e-20);coeff[k]=(beta,ym-beta*xm[ix])
            np.savez(M.OUT/f'B{b}-B{b+1}-predictor.npz',parent=parent.cpu().numpy(),slope=slope.cpu().numpy(),
                intercept=intercept.cpu().numpy(),discovery_correlation=best.cpu().numpy())
            results={}
            for s,(a,z) in {'confirm':(4096,6144),'holdout':(6144,8192)}.items():
                xx=torch.from_numpy(C.unbits(xraw[a:z])).cuda();yy=torch.from_numpy(C.unbits(yraw[a:z])).cuda()
                pred=xx[:,parent]*slope+intercept;mse=((pred-yy)**2).mean(0);base=((yy-ym)**2).mean(0)
                xp=xx[:,parent]-xx[:,parent].mean(0);yp=yy-yy.mean(0)
                rho=(xp*yp).mean(0)/(xp.square().mean(0)*yp.square().mean(0)).sqrt().clamp_min(1e-20)
                results[s]={k:v.cpu().tolist() for k,v in [('mse',mse),('mean_mse',base),('correlation',rho),('ratio',mse/base.clamp_min(1e-20))]}
                for k,ix in controls.items():
                    beta,alpha=coeff[k];results[s][k+'_mse']=((xx[:,ix]*beta+alpha-yy)**2).mean(0).cpu().tolist()
                del xx,yy,pred,xp,yp
            M.write(M.OUT/f'B{b}-B{b+1}-prediction.json',dict(source=b,target=b+1,parents=parent.cpu().tolist(),
                discovery_correlation=best.cpu().tolist(),splits=results,causal_confirmed=False))
            report.append(dict(source=b,target=b+1,holdout_mean_ratio=float(np.mean(results['holdout']['ratio'])),
                holdout_total_MSE_ratio=float(np.sum(results['holdout']['mse'])/max(np.sum(results['holdout']['mean_mse']),1e-20)),
                holdout_better_than_mean=int(np.sum(np.asarray(results['holdout']['ratio'])<.9))))
            del x,y,xm,ym,xs,ys,nx,ny,best,parent,results,corr;gc.collect();torch.cuda.empty_cache()
            M.log('adjacent_boundaries',b-17,10,start)
    rollout=C.unbits(data(18,'A')[6144:]);rows=[]
    for b in M.BLOCKS[:-1]:
        with np.load(M.OUT/f'B{b}-B{b+1}-predictor.npz') as p:
            rollout=rollout[:,p['parent']]*p['slope']+p['intercept']
        truth=C.unbits(data(b+1,'A')[6144:]);mean=C.unbits(data(b+1,'A')[:4096]).mean(0)
        mse=((rollout-truth)**2).mean(0);base=((truth-mean)**2).mean(0)
        ratio=mse/np.maximum(base,1e-20)
        rows.append(dict(target=b+1,mean_mse_ratio=float(ratio.mean()),total_MSE_ratio=float(mse.sum()/max(base.sum(),1e-20)),
            unit_ratios=ratio.astype(float).tolist()))
    M.write(M.OUT/'neighbor-summary.json',dict(one_step=report,rollout=rows,
        scope='All source units searched. Free rollout starts at observed B18 A, not text. Correlation is not a causal edge.'))


def readout():
    import qwen14_mlp_analysis as A
    C.save_json=M.write;A.readout()


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=M.SPECS)
    p.add_argument('stage',choices=['profiles','regimes','predict','neighbors','readout'])
    a=p.parse_args();M.configure(a.model);M.require_admission()
    globals()[a.stage]()


if __name__=='__main__':main()
