"""Secondary, discovery-standardized text prediction with matched controls.

Historical response results motivated this specification. This is a separately
frozen descriptive comparison, never a retrospective replacement of v1 gates.
"""
import argparse
import gc
import time
import numpy as np
import qwen14_goodfire_extension as E
C=E.C


def freeze():
    p=E.OUT/'scaled-prediction-plan.json'
    assert not p.exists()
    E.save(p,{'schema':'standardized_text_prediction_plan.v1','source_sha256':C.file_sha(__file__),
        'motivation':'Historical unscaled frequency features have small numerical scale relative to one-hot token features; old 0-candidate result already known.',
        'exposure':'Historical B28-B35 responses previously exposed; new B36-B39 responses not analyzed at specification. Same corpus is not globally blind.',
        'fit':'Train rows 0:4096 only. Standardize every nonconstant column using discovery mean and population std; intercept left at one.',
        'feature_budget':262,'ridge_per_sample':.01,'seed':20261113,
        'methods':['256 current-token hash buckets + same background (lexical)',
                   '128 current-token +128 preceding-frequency buckets +same background (conditional)',
                   '128 current-token +same background (matched-current control, zero padded)',
                   'conditional with discovery target documents cyclically shifted137 (negative control)'],
        'criteria':'Both confirm and holdout conditional MSE <=0.9*lexical; conditional beats mean, matched-current and shifted-target controls. Report all coordinates and failures.',
        'scope':'Secondary descriptive continuous prediction screen; not FWER-controlled semantic confirmation, not behavioral accuracy.'})


def run():
    import torch
    from threadpoolctl import threadpool_limits
    p=C.read(E.OUT/'scaled-prediction-plan.json');assert p['source_sha256']==C.file_sha(__file__)
    threadpool_limits(4);torch.set_num_threads(4)
    features=np.load(E.OUT/'text-features.npz');X={k:features[k].astype(np.float64) for k in ['baseline','conditional']}
    X['matched_current']=X['conditional'].copy();X['matched_current'][:,128:256]=0
    ops={};scales={}
    for key,x in X.items():
        mean=x[:4096].mean(0);std=x[:4096].std(0);keep=std>1e-12
        mean[256]=0;std[256]=1;keep[256]=True
        x[:,keep]=(x[:,keep]-mean[keep])/std[keep];x[:,~keep]=0
        penalty=np.eye(262)*p['ridge_per_sample']*4096;penalty[256,256]=0
        ops[key]=np.linalg.solve(x[:4096].T@x[:4096]+penalty,x[:4096].T).astype(np.float32)
        scales[key]={'mean':mean.tolist(),'std':std.tolist(),'nonconstant':keep.tolist()};X[key]=x.astype(np.float32)
    E.save(E.OUT/'scaled-prediction-feature-scales.json',scales)
    summary=[];start=time.monotonic()
    with C.GpuLease(),torch.inference_mode():
        tx={k:torch.from_numpy(x).cuda() for k,x in X.items()};op={k:torch.from_numpy(x).cuda() for k,x in ops.items()}
        for b in E.ALL:
            arrays={};counts={}
            for field in ['G','V','A']:
                raw=E.data(b,field)
                values={s:{k:[] for k in ['mean','baseline','conditional','matched_current','shuffled']} for s in ['confirm','holdout']}
                coefficient=np.lib.format.open_memmap(E.OUT/f'B{b}-{field}-scaled-coeff.npy',mode='w+',dtype=np.float32,shape=(262,C.WIDTH))
                for lo in range(0,C.WIDTH,256):
                    y=torch.from_numpy(C.unbits(raw[:,lo:lo+256])).cuda();train=y[:4096]
                    weights={k:o@train for k,o in op.items()};coefficient[:,lo:lo+256]=weights['conditional'].cpu().numpy()
                    wrong=op['conditional']@torch.roll(train.reshape(512,8,-1),137,0).reshape(4096,-1)
                    for s,(a,z) in {'confirm':(4096,6144),'holdout':(6144,8192)}.items():
                        truth=y[a:z]
                        predictions={'mean':train.mean(0),'shuffled':tx['conditional'][a:z]@wrong,
                            **{k:tx[k][a:z]@weights[k] for k in weights}}
                        for k,v in predictions.items():values[s][k].extend(((truth-v)**2).mean(0).cpu().numpy())
                coefficient.flush();del coefficient
                good=np.ones(C.WIDTH,dtype=bool)
                for s,v in values.items():
                    v={k:np.asarray(x) for k,x in v.items()}
                    for k,x in v.items():arrays[field+'_'+s+'_'+k+'_mse']=x
                    good&=(v['conditional']<=.9*v['baseline'])
                    for k in ['mean','matched_current','shuffled']:good&=v['conditional']<v[k]
                arrays[field+'_candidate']=good;counts[field]=int(good.sum())
                E.log('scaled_text_prediction',(b-28)*3+['G','V','A'].index(field)+1,36,start,block=b,field=field)
            np.savez_compressed(E.OUT/f'B{b}-scaled-prediction.npz',**arrays)
            summary.append({'block':b,'candidates':counts,'mean_holdout_MSE_ratio':{f:float(np.mean(arrays[f+'_holdout_conditional_mse']/np.maximum(arrays[f+'_holdout_baseline_mse'],1e-20))) for f in ['G','V','A']}})
            gc.collect();torch.cuda.empty_cache()
    E.save(E.OUT/'scaled-prediction-summary.json',{'layers':summary,'plan_sha256':C.file_sha(E.OUT/'scaled-prediction-plan.json'),
        'scope':'Secondary descriptive screen with discovery-only feature scaling; does not revise historical negative results or assign semantic labels.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    E.configure();globals()[args.stage]()
