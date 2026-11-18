"""All-coordinate signed-regime audit and frozen adjacent-layer prediction."""
import argparse
import gc
import time
import numpy as np
import qwen14_goodfire_extension as E
C=E.C


def regimes():
    start=time.monotonic();summary=[]
    for b in E.ALL:
        maps={f:E.data(b,f) for f in ['G','V','A']};arrays={}
        for split,(low,high) in C.SPLITS.items():
            lo,hi=low*8,high*8
            out={k:[] for k in ['quadrant_fraction','quadrant_abs_mass_fraction','quadrant_energy_fraction',
                               'negative_gate_fraction','negative_silu_derivative_fraction','A_mean','A_std']}
            extrema={k:[] for k in ['double_negative_max_index','double_negative_max_value',
                                   'double_positive_max_index','double_positive_max_value']}
            for u in range(0,C.WIDTH,256):
                z=min(C.WIDTH,u+256)
                g,v,a=[C.unbits(maps[f][lo:hi,u:z]) for f in ['G','V','A']]
                abs_a=np.abs(a).astype(np.float64);sq_a=a.astype(np.float64)**2
                quad=[(g>=0)&(v>=0),(g>=0)&(v<0),(g<0)&(v>=0),(g<0)&(v<0)]
                out['quadrant_fraction'].extend(np.stack([m.mean(0) for m in quad],1))
                out['quadrant_abs_mass_fraction'].extend(np.stack([(abs_a*m).sum(0)/np.maximum(abs_a.sum(0),1e-30) for m in quad],1))
                out['quadrant_energy_fraction'].extend(np.stack([(sq_a*m).sum(0)/np.maximum(sq_a.sum(0),1e-30) for m in quad],1))
                # Stable SiLU derivative; branch is nonmonotone for sufficiently negative G.
                sig=1/(1+np.exp(-np.clip(g,-80,80)))
                deriv=sig*(1+g*(1-sig))
                out['negative_gate_fraction'].extend((g<0).mean(0))
                out['negative_silu_derivative_fraction'].extend((deriv<0).mean(0))
                out['A_mean'].extend(a.mean(0,dtype=np.float64));out['A_std'].extend(a.std(0,dtype=np.float64))
                for label,m in [('double_negative',quad[3]),('double_positive',quad[0])]:
                    masked=np.where(m,a,-np.inf);ix=masked.argmax(0);val=masked[ix,np.arange(z-u)]
                    ix=np.where(np.isfinite(val),ix+lo,-1)
                    extrema[label+'_max_index'].extend(ix)
                    extrema[label+'_max_value'].extend(np.where(np.isfinite(val),val,0))
            arrays.update({split+'_'+k:np.asarray(v) for k,v in {**out,**extrema}.items()})
        np.savez_compressed(E.OUT/f'B{b}-regimes.npz',**arrays)
        h=arrays['holdout_quadrant_abs_mass_fraction'][:,3]
        summary.append({'block':b,'units':C.WIDTH,'holdout_DN_mass_mean':float(h.mean()),
            'holdout_DN_mass_median':float(np.median(h)), 'holdout_DN_mass_over_10pct':int((h>.1).sum()),
            'holdout_DN_mass_over_50pct':int((h>.5).sum()),
            'holdout_negative_silu_derivative_mean':float(arrays['holdout_negative_silu_derivative_fraction'].mean())})
        E.log('signed_regimes',b-27,12,start,block=b)
        del maps,arrays;gc.collect()
    E.save(E.OUT/'regime-summary.json',{'layers':summary,'quadrants':['G>=0,V>=0','G>=0,V<0','G<0,V>=0','G<0,V<0'],
        'scope':'Observed signed arithmetic, no semantic hypothesis or statistical confirmation. Zero G/V assigned to nonnegative quadrant.'})


def neighbors():
    import torch
    start=time.monotonic();report=[]
    with C.GpuLease(),torch.inference_mode():
        torch.set_num_threads(4)
        for b in range(35,39):
            xraw=E.data(b,'A');yraw=E.data(b+1,'A')
            x=torch.from_numpy(C.unbits(xraw[:4096])).cuda();y=torch.from_numpy(C.unbits(yraw[:4096])).cuda()
            xm=x.mean(0);ym=y.mean(0);x-=xm;y-=ym
            xs=x.square().mean(0).sqrt();ys=y.square().mean(0).sqrt()
            nx=x/xs.clamp_min(1e-12);ny=y/ys.clamp_min(1e-12)
            best=torch.zeros(C.WIDTH,device='cuda');parent=torch.zeros(C.WIDTH,dtype=torch.long,device='cuda')
            for lo in range(0,C.WIDTH,256):
                corr=nx[:,lo:lo+256].T@ny/4096
                values,ix=corr.abs().max(0);signed=corr[ix,torch.arange(C.WIDTH,device='cuda')]
                choose=values>best.abs();best[choose]=signed[choose];parent[choose]=ix[choose]+lo
            slope=best*ys/xs[parent].clamp_min(1e-12);intercept=ym-slope*xm[parent]
            controls={'same_index':torch.arange(C.WIDTH,device='cuda'),
                'random_parent':torch.from_numpy(np.random.default_rng(20261109+b).integers(0,C.WIDTH,C.WIDTH)).cuda()}
            coeff={}
            for k,ix in controls.items():
                beta=(x[:,ix]*y).mean(0)/xs[ix].square().clamp_min(1e-20)
                coeff[k]=(beta,ym-beta*xm[ix])
            np.savez(E.OUT/f'B{b}-B{b+1}-predictor.npz',parent=parent.cpu().numpy(),slope=slope.cpu().numpy(),
                intercept=intercept.cpu().numpy(),discovery_correlation=best.cpu().numpy())
            results={}
            for s,(a,z) in {'confirm':(4096,6144),'holdout':(6144,8192)}.items():
                xx=torch.from_numpy(C.unbits(xraw[a:z])).cuda();yy=torch.from_numpy(C.unbits(yraw[a:z])).cuda()
                pred=xx[:,parent]*slope+intercept
                mse=((pred-yy)**2).mean(0);base=((yy-ym)**2).mean(0)
                xp=xx[:,parent]-xx[:,parent].mean(0);yp=yy-yy.mean(0)
                rho=(xp*yp).mean(0)/(xp.square().mean(0)*yp.square().mean(0)).sqrt().clamp_min(1e-20)
                results[s]={k:v.cpu().numpy().astype(float).tolist() for k,v in
                    [('mse',mse),('mean_mse',base),('correlation',rho),('ratio',mse/base.clamp_min(1e-20))]}
                for k,ix in controls.items():
                    beta,alpha=coeff[k];results[s][k+'_mse']=((xx[:,ix]*beta+alpha-yy)**2).mean(0).cpu().tolist()
                del xx,yy,pred,xp,yp
            E.save(E.OUT/f'B{b}-B{b+1}-prediction.json',{'source':b,'target':b+1,'parents':parent.cpu().tolist(),
                'discovery_correlation':best.cpu().tolist(),'splits':results,'causal_confirmed':False})
            report.append({'source':b,'target':b+1,'holdout_mean_ratio':float(np.mean(results['holdout']['ratio'])),
                'holdout_better_than_mean':int(np.sum(np.asarray(results['holdout']['ratio'])<.9))})
            del x,y,xm,ym,xs,ys,nx,ny,best,parent,results,corr;gc.collect();torch.cuda.empty_cache()
            E.log('new_adjacent_boundaries',b-34,4,start)
    # Combined trajectory uses old frozen estimators through B35, then new ones.
    rollout=C.unbits(E.data(28,'A')[6144:]);rows=[]
    for b in range(28,39):
        root=E.OLD if b<35 else E.OUT
        with np.load(root/f'B{b}-B{b+1}-predictor.npz') as p:
            rollout=rollout[:,p['parent']]*p['slope']+p['intercept']
        truth=C.unbits(E.data(b+1,'A')[6144:]);mean=C.unbits(E.data(b+1,'A')[:4096]).mean(0)
        ratio=((rollout-truth)**2).mean(0)/np.maximum(((truth-mean)**2).mean(0),1e-20)
        rows.append({'target':b+1,'mean_mse_ratio':float(ratio.mean()),'unit_ratios':ratio.astype(float).tolist()})
    E.save(E.OUT/'neighbor-summary.json',{'one_step':C.read(E.OLD/'neighbor-summary.json')['one_step']+report,'rollout':rows,
        'scope':'One observed upstream unit predicts each target; fit discovery only. All pairs considered. Correlation is not a causal edge. Complete rollout starts from observed B28 A.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['regimes','neighbors']);args=parser.parse_args()
    E.configure();globals()[args.stage]()
