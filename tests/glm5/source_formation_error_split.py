"""Diagnostic Q-only/K-only oracle replays and centered bilinear error accounting."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_behavior import Runtime
from source_formation_analysis import predict_fit,rotary,rotate_qk,attention,metric,close
from source_binding_native_v2 import unbits

def prereg():
    assert (D.OUT/'qwen14/confirm-formation.json').exists()
    assert not (D.OUT/'qwen14/lockbox_frame-behavior.json').exists()
    D.save('formation-error-split-addendum.json',{'code_sha256':D.sha(Path(__file__)),
        'exposure':'Qwen confirm formation and update output have been viewed; this is a posthoc diagnostic there. The same recipe is frozen before new-frame behavior or internal results. No primary candidates, coefficients, thresholds or GLM gate changed.',
        'oracle_controls':['true current Q plus predicted K','predicted Q plus true current K'],
        'algebra':'Predicted score minus actual score = dQ dot K + Q dot dK + dQ dot dK, each term centered across valid prefix tokens per row to remove softmax-invariant additive offsets. Report full3x3 second-moment matrix; covariance terms are not causal shares.',
        'meaning':'Localize forecast errors with privileged current-layer Q or K; not usable early-state predictors or interventions.'})
def evaluate(model,split):
    from source_formation_native import verify
    verify();D.admission(model,split);a=D.read(D.OUT/'formation-error-split-addendum.json');assert a['code_sha256']==D.sha(Path(__file__))
    root=D.OUT/model;meta=D.read(root/'formation-fit.json');params=np.load(root/'formation-fit.npz');rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    manifest=D.read(root/f'{split}-native-manifest.json');L=manifest['length'];n=len(rows);rt=Runtime(model);timer=time.monotonic()
    arrays={x:np.load(root/f'{split}-native/{x}.npy',mmap_mode='r') for x in ['early','qpre','kpre','vproj','head_context']}
    results=[]
    for bi,c in enumerate(meta['heads']):
        tag=f'B{c["block"]}H{c["head"]}';w=rt.weight(f'model.layers.{c["block"]}.self_attn.o_proj.weight','cpu')[:,c['head']*rt.hd:(c['head']+1)*rt.hd].float().numpy().copy()
        truth=unbits(arrays['head_context'][:,bi,c['head']])@w.T;mean=params[f'B{c["block"]}_mean_context']@w.T
        predicted={'true_Q_pred_K':[],'pred_Q_true_K':[]};cov=np.zeros((3,3));tokens=0;identity_errors=[];qcos=[];qnorm_ratio=[]
        for lo in range(0,n,32):
            hi=min(lo+32,n);rs=rows[lo:hi];x=unbits(arrays['early'][lo:hi])
            qp=predict_fit(params,'q_native',x[:,-1])[:,bi*rt.hd:(bi+1)*rt.hd]
            kp=predict_fit(params,'k_native',x)[:,:,bi*rt.hd:(bi+1)*rt.hd]
            q=unbits(arrays['qpre'][lo:hi,bi,c['head']]);k=unbits(arrays['kpre'][lo:hi,bi,c['kv_head']]);v=unbits(arrays['vproj'][lo:hi,bi,c['kv_head']])
            pe=rotary(rt,rs,L);lengths=[len(r['tokens']) for r in rs]
            for name,qq,kk in [('true_Q_pred_K',q,kp),('pred_Q_true_K',qp,k)]:
                context,_=attention(rt,qq,kk,v,pe,lengths);predicted[name].append(context@w.T)
            qr,kr=rotate_qk(rt,q,k,pe);qpr,kpr=rotate_qk(rt,qp,kp,pe)
            qr,kr,qpr,kpr=[z.astype(np.float64) for z in [qr,kr,qpr,kpr]]
            dq=qpr-qr;dk=kpr-kr;scale=rt.hd**-.5
            terms=np.stack([np.einsum('nd,ntd->nt',dq,kr)*scale,np.einsum('nd,ntd->nt',qr,dk)*scale,np.einsum('nd,ntd->nt',dq,dk)*scale],axis=-1)
            difference=(np.einsum('nd,ntd->nt',qpr,kpr)-np.einsum('nd,ntd->nt',qr,kr))*scale
            for i,size in enumerate(lengths):
                t=terms[i,-size:];e=difference[i,-size:];t=t-t.mean(0);e=e-e.mean()
                identity_errors.append(float(np.max(np.abs(t.sum(1)-e))))
                cov+=t.T@t;tokens+=size
            qcos.extend((np.sum(q*qp,axis=1)/np.maximum(np.linalg.norm(q,axis=1)*np.linalg.norm(qp,axis=1),1e-12)).tolist())
            qnorm_ratio.extend((np.linalg.norm(qp,axis=1)/np.maximum(np.linalg.norm(q,axis=1),1e-12)).tolist())
            del x,q,k,v,qr,kr,qpr,kpr,terms;gc.collect()
        moments=cov/tokens
        assert max(identity_errors)<1e-8
        entry={'id':tag,'methods':{name:metric(rows,truth,np.concatenate(values),mean,f'{model}/{split}/{tag}/{name}') for name,values in predicted.items()},
            'centered_score_error_second_moment_3x3':moments.tolist(),'centered_total_score_error_MSE':float(moments.sum()),
            'term_order':['dQ_dot_K','Q_dot_dK','dQ_dot_dK'],'algebra_identity_max_abs_error':max(identity_errors),
            'query_cosine_mean':float(np.mean(qcos)),'query_predicted_over_true_norm_mean':float(np.mean(qnorm_ratio))}
        results.append(entry);D.log(model+'/'+split+'/error-split',bi+1,4,timer)
    close(*arrays.values());params.close()
    D.save(f'{model}/{split}-error-split.json',{'results':results,'addendum_sha256':D.sha(D.OUT/'formation-error-split-addendum.json'),
        'evidence_status':'Qwen confirm posthoc; recipe frozen before Qwen new-frame/GLM outcomes; privileged-input error diagnostic, not autonomous prediction.'},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prereg','evaluate']);ap.add_argument('--model');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='prereg':prereg()
    else:evaluate(a.model,a.split)
