"""Conditional recurrent updates with forecast routing and preserved past-token V."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_analysis import ridge,predict,bootstrap,close
from source_binding_native_v2 import unbits
from source_formation_native import verify
METHODS=['mean','state128','routed_content','shuffled_content']

def fit(model):
    verify();D.admission(model,'discovery');root=D.OUT/model;meta=D.read(root/'formation-fit.json')
    h=np.load(root/'discovery-native/hidden.npy',mmap_mode='r');writes=np.load(root/'discovery-formation-predictions.npz')
    byblock={c['block']:f'B{c["block"]}H{c["head"]}' for c in meta['heads']}
    params={};timer=time.monotonic();dim=h.shape[-1]
    for b in range(D.EARLY,40):
        x=unbits(h[:,b,-1]);y=unbits(h[:,b+1,-1])-x
        idx=np.sort(np.argsort(x.var(0),kind='stable')[-128:]);params[f'B{b}_idx']=idx;params[f'B{b}_mean']=y.mean(0)
        for method in ['state128','routed_content']:
            target=y-writes[byblock[b]+'_native_past'] if method=='routed_content' and b in byblock else y
            for key,v in ridge(x[:,idx],target).items():params[f'B{b}_{method}_{key}']=v
        D.log(model+'/update-fit',b-D.EARLY+1,40-D.EARLY,timer)
    close(h);writes.close();path=root/'update-fit.npz';assert not path.exists();np.savez(path,**params)
    D.save(f'{model}/update-fit.json',{'fit_sha256':D.sha(path),'formation_fit_sha256':meta['fit_sha256'],'start_boundary':D.EARLY,
        'input_budget':'Both remainder fits use128 native coordinates plus intercept per layer. Routed adds learned QK forecasts, four original WO column blocks and original past-token V. Not equal total information or computation budget.',
        'remainder_coefficients_per_method':20*129*dim,
        'no_future_last_state':'Forecast QK from actual hs20 once; each conditional write excludes final query V. Subsequent state predictors use own current predictions.',
        'control':'shuffled_content uses same routed remainder coefficients, cyclically permutes six record V segments within each example; source/entity contextual information can move too.',
        'prereg':'Frozen before first new internal observation; no adaptation to holdout.'},model)

def evaluate(model,split):
    verify();D.admission(model,split);root=D.OUT/model;info=D.read(root/'update-fit.json');params=np.load(root/'update-fit.npz')
    assert info['fit_sha256']==D.sha(root/'update-fit.npz')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];meta=D.read(root/'formation-fit.json')
    byblock={c['block']:f'B{c["block"]}H{c["head"]}' for c in meta['heads']}
    h=np.load(root/f'{split}-native/hidden.npy',mmap_mode='r');writes=np.load(root/f'{split}-formation-predictions.npz')
    states={m:unbits(h[:,D.EARLY,-1]) for m in METHODS};steps=[];errors=[];timer=time.monotonic()
    for b in range(D.EARLY,40):
        actual=unbits(h[:,b,-1]);target=unbits(h[:,b+1,-1]);dy=target-actual;idx=params[f'B{b}_idx'];base=np.square(dy-params[f'B{b}_mean']).sum(1)
        metrics={};se={}
        for m in METHODS:
            if m=='mean':one=np.broadcast_to(params[f'B{b}_mean'],dy.shape);states[m]+=one
            else:
                use='state128' if m=='state128' else 'routed_content'
                f={k:params[f'B{b}_{use}_{k}'] for k in ['center','scale','coef']}
                w=0
                if m!='state128' and b in byblock:w=writes[byblock[b]+('_native_past_shuffled' if m=='shuffled_content' else '_native_past')]
                one=predict(actual[:,idx],f)+w
                states[m]+=predict(states[m][:,idx],f)+w
            error=np.square(target-states[m]).sum(1);se[m]=error
            metrics[m]={'one_step_error_vs_mean':float(np.square(dy-one).sum()/max(base.sum(),1e-12)),
                'rollout_MSE':float(error.mean()),'rollout_relative_error':float(np.sqrt(error.sum()/max(np.square(target).sum(),1e-12)))}
        errors.append(np.stack([se[m] for m in METHODS],axis=1));steps.append({'block':b,'methods':metrics})
        D.log(model+'/'+split+'/update',b-D.EARLY+1,40-D.EARLY,timer)
    close(h);writes.close();gain=errors[-1][:,1]-errors[-1][:,2]
    path=root/f'{split}-update-final.npz';assert not path.exists();np.savez(path,**states,per_row_step_errors=np.stack(errors,axis=1))
    D.save(f'{model}/{split}-update.json',{'steps':steps,'routed_vs_state_MSE_gain':float(gain.mean()),
        'routed_gain_ci_family':bootstrap(rows,gain,f'{model}/{split}/updategain'),'final_sha256':D.sha(path),
        'fit_sha256':info['fit_sha256'],'scope':'Conditional recursion with real prefix memory. No true last-token states after hs20; original earlier-token V still supplied.'},model)

def output(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from source_formation_behavior import Runtime
    verify();D.admission(model,split);root=D.OUT/model;rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
    data=np.load(root/f'{split}-update-final.npz');h=np.load(root/f'{split}-native/hidden.npy',mmap_mode='r')
    base=D.read(root/f'{split}-behavior.json')['rows'];saved={m:[] for m in METHODS};timer=time.monotonic()
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model)
        for lo in range(0,len(rows),128):
            hi=min(lo+128,len(rows));rs=rows[lo:hi];raw=torch.from_numpy(np.array(h[lo:hi,40,-1])).view(torch.bfloat16).cuda()
            _,lg=rt.finish(raw);scored=rt.score(lg,rs)
            assert all(x['top1']==y['top1'] and x['scores']==y['scores'] for x,y in zip(scored,base[lo:hi]))
            lp=lg.log_softmax(-1);p=lp.exp();top=lg.argmax(-1)
            for m in METHODS:
                state=torch.from_numpy(np.array(data[m][lo:hi])).cuda().bfloat16();_,pred=rt.finish(state);kl=(p*(lp-pred.log_softmax(-1))).sum(-1)
                items=rt.score(pred,rs);equal=pred.argmax(-1)==top
                for i,r in enumerate(items):r.update(KL=float(kl[i]),top1_matches_model=bool(equal[i]))
                saved[m]+=items;del state,pred
            del raw,lg,lp,p,top;gc.collect();torch.cuda.empty_cache();D.log(model+'/'+split+'/update-output',hi,len(rows),timer)
    close(h);data.close()
    summary={m:{'KL_mean':float(np.mean([x['KL'] for x in rs])),'top1_agreement':float(np.mean([x['top1_matches_model'] for x in rs])),
        'forced_accuracy':float(np.mean([x['forced_correct'] for x in rs]))} for m,rs in saved.items()}
    D.save(f'{model}/{split}-update-output.json',{'summary':summary,'rows':saved,'actual_final_logits_replay_exact':True},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate','output']);ap.add_argument('--model',required=True);ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='fit':fit(a.model)
    else:globals()[a.stage](a.model,a.split)
