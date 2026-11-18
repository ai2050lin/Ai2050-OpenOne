"""Exploratory last-token rollout with a strictly past-token KV memory input."""
import argparse,sys,time,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_native import unbits,bits
from source_binding_analysis import verify_code,bootstrap

METHODS=['layer_mean','h32','h16_prefix16']

def prereg():
    D.save('conditional-update-addendum.json',{'code_sha256':D.sha(Path(__file__)),'original_plan_sha256':D.sha(D.OUT/'plan.json'),
        'status':'Qwen exploratory after component holdouts opened; recipe fixed before GLM pilot. Does not change primary gates or replication decision.',
        'hypothesis':'A32feature linear update may improve with16 current last-state projections plus16 ordered-prefixKV projections versus32 last-state projections; test rather than assume.',
        'inputs':'Actual start hidden state once. At selected head blocks only, earlier-token K/V (excluding last query) from original prefix computation. Other blocks use h32 for both methods.',
        'prefix_projection':'fixed Rademacher weights by logical prefix position, native KV coordinate and K/V kind; all past tokens; sqrt(length) normalization; no answer labels',
        'fit':'discovery only; per-layer standardization; ridge lambda1,32 features plus intercept; no hyperparameter or start-layer search',
        'start':'minimum frozen candidate-head block; roll through raw boundary40; all original hidden coordinates scored',
        'baselines':'discovery layer-mean increment; h32; h16+prefix16 (same33 output coefficients per coordinate per layer)',
        'scope':'one-step with true current state and genuinely recursive predicted last state; prefix memory remains original and costs an original prefix computation, so not a text-only or full-field autonomous model',
        'evaluation':'all rows, raw full-state errors and original-unembedding output KL; descriptive, no new confirmatory threshold',
        'formal_evidence_promotion':0})

def verify():
    verify_code();a=D.read(D.OUT/'conditional-update-addendum.json');assert a['code_sha256']==D.sha(Path(__file__))

def projection(model,dim):return D.rng(f'{model}/update/h32').choice(np.array([-1,1],np.float32),(dim,32))/np.sqrt(dim)

def prefix_features(model,split,block):
    root=D.OUT/model;m=D.read(root/f'{split}-native-manifest.json');bi=m['head_blocks'].index(block)
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];lengths=np.array([len(r['tokens']) for r in rows]);n=len(rows)
    k=np.load(root/f'{split}-native/key.npy',mmap_mode='r');v=np.load(root/f'{split}-native/vproj.npy',mmap_mode='r')
    dim=k.shape[2]*k.shape[-1];z=np.zeros((n,16),np.float32);scale=np.sqrt(2*dim)
    for pos in range(int(lengths.max())-1):
        ii=np.flatnonzero(pos<lengths-1);jj=m['length']-lengths[ii]+pos
        # Last-token K/V are never included, also during recursive rollout.
        assert np.all(jj<m['length']-1)
        for kind,array in [('k',k),('v',v)]:
            w=D.rng(f'{model}/update/prefix/{kind}/{pos}').choice(np.array([-1,1],np.float32),(dim,16))/scale
            x=unbits(array[ii,bi,:,jj,:]).reshape(len(ii),dim);z[ii]+=x@w
    k._mmap.close();v._mmap.close()
    return z/np.sqrt(lengths[:,None]-1)

def ridge(x,y):
    mean=x.mean(0);std=x.std(0);std=np.maximum(std,1e-6);a=np.c_[np.ones(len(x)),(x-mean)/std]
    penalty=np.eye(a.shape[1]);penalty[0,0]=0;coef=np.linalg.solve(a.T@a+penalty,a.T@y)
    return mean.astype('f'),std.astype('f'),coef.astype('f')

def predict(x,fit):
    mean,std,coef=fit;return coef[0]+((x-mean)/std)@coef[1:]

def fit(model):
    verify();D.admission(model,'discovery');root=D.OUT/model;c=D.read(root/'candidates.json');blocks=sorted(x['block'] for x in c['heads']);start=blocks[0]
    h=np.load(root/'discovery-native/hidden.npy',mmap_mode='r');p=projection(model,h.shape[-1]);saved={'projection':p};timer=time.monotonic()
    for b in range(start,40):
        x=unbits(h[:,b,-1]);y=unbits(h[:,b+1,-1])-x;saved[f'B{b}_mean']=y.mean(0);xh=x@p
        for method in ['h32','h16_prefix16']:
            xx=xh if method=='h32' or b not in blocks else np.c_[xh[:,:16],prefix_features(model,'discovery',b)]
            mean,std,coef=ridge(xx,y)
            for name,val in [('center',mean),('scale',std),('coef',coef)]:saved[f'B{b}_{method}_{name}']=val
        D.log(model+'/conditional-update-fit',b-start+1,40-start,timer)
    path=root/'conditional-update-fit.npz';assert not path.exists();np.savez(path,**saved)
    D.save(f'{model}/conditional-update-fit.json',{'start_boundary':start,'memory_blocks':blocks,'fit_sha256':D.sha(path),
        'addendum_sha256':D.sha(D.OUT/'conditional-update-addendum.json'),'candidate_sha256':D.sha(root/'candidates.json'),
        'discovery_manifest_sha256':D.sha(root/'discovery-native-manifest.json'),'linear_coefficients_per_method':(40-start)*h.shape[-1]*33},model)

def evaluate(model,split):
    verify();D.admission(model,split);root=D.OUT/model;info=D.read(root/'conditional-update-fit.json');params=np.load(root/'conditional-update-fit.npz')
    assert info['fit_sha256']==D.sha(root/'conditional-update-fit.npz')
    h=np.load(root/f'{split}-native/hidden.npy',mmap_mode='r');rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    start=info['start_boundary'];p=params['projection'];states={name:unbits(h[:,start,-1]) for name in METHODS};errors=[];stepmetrics=[];timer=time.monotonic()
    for b in range(start,40):
        actual=unbits(h[:,b,-1]);target=unbits(h[:,b+1,-1]);dy=target-actual;mean=params[f'B{b}_mean'];base=np.square(dy-mean).sum(1)
        kv=prefix_features(model,split,b) if b in info['memory_blocks'] else None
        metrics={};state_errors={}
        for method in METHODS:
            if method=='layer_mean':prediction=np.broadcast_to(mean,dy.shape);states[method]+=mean
            else:
                fit=tuple(params[f'B{b}_{method}_{name}'] for name in ['center','scale','coef'])
                xh=actual@p;xx=xh if method=='h32' or kv is None else np.c_[xh[:,:16],kv];prediction=predict(xx,fit)
                xh=states[method]@p;xx=xh if method=='h32' or kv is None else np.c_[xh[:,:16],kv];states[method]+=predict(xx,fit)
            se=np.square(dy-prediction).sum(1);rs=np.square(target-states[method]).sum(1);state_errors[method]=rs
            metrics[method]={'one_step_error_vs_mean':float(se.sum()/base.sum()),'one_step_squared_error':float(se.mean()),
                'rollout_squared_error':float(rs.mean()),'rollout_whole_state_relative_error':float(np.sqrt(rs.sum()/np.square(target).sum()))}
        stepmetrics.append({'block':b,'methods':metrics});errors.append(np.stack([state_errors[m] for m in METHODS],axis=-1))
        D.log(model+'/'+split+'/conditional-update-eval',b-start+1,40-start,timer)
    path=root/f'{split}-conditional-update-final.npz';assert not path.exists();np.savez(path,**states,per_row_rollout_squared_error=np.stack(errors,axis=1))
    gain=np.square(unbits(h[:,40,-1])-states['h32']).sum(1)-np.square(unbits(h[:,40,-1])-states['h16_prefix16']).sum(1)
    D.save(f'{model}/{split}-conditional-update.json',{'start_boundary':start,'memory_blocks':info['memory_blocks'],'steps':stepmetrics,
        'prefix_vs_h32_squared_error_gain':float(gain.mean()),'prefix_vs_h32_gain_ci_family':bootstrap(rows,gain,f'{model}/{split}/prefixGain'),
        'final_sha256':D.sha(path),'fit_sha256':info['fit_sha256'],'addendum_sha256':D.sha(D.OUT/'conditional-update-addendum.json'),
        'evidence_status':'exploratory for Qwen; fixed before GLM execution; no new formal confirmation'},model)

def output(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from source_binding_runtime import Runtime
    verify();D.admission(model,split);root=D.OUT/model;data=np.load(root/f'{split}-conditional-update-final.npz');rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    raw=np.load(root/f'{split}-native/hidden.npy',mmap_mode='r');behavior=D.read(root/f'{split}-behavior.json')['rows'];saved={m:[] for m in METHODS};checks=[];timer=time.monotonic()
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model)
        for lo in range(0,len(rows),128):
            hi=min(lo+128,len(rows));rs=rows[lo:hi];real=torch.from_numpy(np.array(raw[lo:hi,40,-1])).view(torch.bfloat16).cuda()
            _,lg=rt.finish(real);base=rt.score(lg,rs);diff=max(abs(a-b) for x,y in zip(base,behavior[lo:hi]) for a,b in zip(x['scores'],y['scores']))
            assert diff==0 and all(x['top1']==y['top1'] for x,y in zip(base,behavior[lo:hi]));lp=lg.log_softmax(-1);prob=lp.exp();top=lg.argmax(-1)
            checks.append({'chunk':lo,'raw_final_replay_exact':True})
            for method in METHODS:
                state=torch.from_numpy(np.array(data[method][lo:hi])).cuda().bfloat16();_,pred=rt.finish(state);kl=(prob*(lp-pred.log_softmax(-1))).sum(-1)
                result=rt.score(pred,rs);match=pred.argmax(-1)==top
                for i,x in enumerate(result):x.update(kl_from_true=float(kl[i]),top1_matches_model=bool(match[i]))
                saved[method]+=result
                del state,pred
            del real,lg,lp,prob,top;gc.collect();torch.cuda.empty_cache();D.log(model+'/'+split+'/conditional-output',hi,len(rows),timer)
    summary={m:{'KL_mean':float(np.mean([r['kl_from_true'] for r in rs])),'top1_agreement':float(np.mean([r['top1_matches_model'] for r in rs])),
                   'forced_accuracy':float(np.mean([r['forced_correct'] for r in rs]))} for m,rs in saved.items()}
    D.save(f'{model}/{split}-conditional-output.json',{'summary':summary,'rows':saved,'checks':checks,
        'input_status':'one true start last-token state, plus original past-token KV at frozen memory blocks for hybrid; not full-field autonomous'},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prereg','fit','evaluate','output']);ap.add_argument('--model',default='qwen14');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='prereg':prereg()
    elif a.stage=='fit':fit(a.model)
    else:globals()[a.stage](a.model,a.split)
