"""Secondary recurrence: predicted content versus true historical content, no late-state feedback."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
import source_relay2_math as M
from source_relay2_fit import load,close
from source_binding_native_v2 import unbits
from source_formation_analysis import ridge,predict
METHODS=['mean_next','state_direct','state_residual','aware_predV_direct','ridge_predV_direct','aware_trueVpast_direct','meanV_direct']
CONTEXT={'aware_predV_direct':'aware_native_predV','ridge_predV_direct':'native_predV',
         'aware_trueVpast_direct':'aware_native_past_trueV','meanV_direct':'aware_native_meanV'}

def fit():
    D.verify_internal();D.admission('discovery');root=D.OUT/D.MODEL;h=load('discovery','hidden');fore=np.load(root/'discovery-forecast.npz');fk=np.load(root/'fit.npz')
    params={};timer=time.monotonic()
    for b in range(20,40):
        x=unbits(h[:,b,-1]);target=unbits(h[:,b+1,-1]);idx=np.sort(np.argsort(x.var(0),kind='stable')[-128:])
        params[f'B{b}_idx']=idx;params[f'B{b}_mean']=target.mean(0)
        for kind in ['direct','residual','remainder']:
            y=target-x if kind=='residual' else target
            if kind=='remainder' and b in [28,29]:
                bi=[28,29].index(b);y=y-fore['truth'][:,bi]@fk[f'wo_{bi}'].T
            for key,val in ridge(x[:,idx],y).items():params[f'B{b}_{kind}_{key}']=val
        D.log('recurrence/fit',b-19,20,timer)
    close(h);fore.close();fk.close();path=root/'update-fit.npz';assert not path.exists();np.savez(path,**params)
    D.save(f'{D.MODEL}/update-fit.json',{'fit_sha256':D.sha(path),'routing_fit_sha256':D.sha(root/'fit.npz'),
        'width':128,'selection':'largest discovery state coordinate variance, identical inputs across direct/residual/remainder fits',
        'recipe':'direct next-state ridge; residual increment ridge; direct remainder subtracts actual two-head writing only during training. Future forecasts from initial hs20 prefix; subsequent state inputs are predicted.',
        'budget':'Routed versions add learned QKV and WO; same remainder width does not make total compute/information budgets equal.'})

def evaluate(split):
    D.verify_internal();D.admission(split);root=D.OUT/D.MODEL;info=D.read(root/'update-fit.json');assert D.sha(root/'update-fit.npz')==info['fit_sha256']
    params=np.load(root/'update-fit.npz');fore=np.load(root/f'{split}-forecast.npz');fk=np.load(root/'fit.npz');h=load(split,'hidden')
    rows=D.read(D.OUT/'qwen14-materials.json')['splits'][split];states={m:unbits(h[:,20,-1]) for m in METHODS};steps=[];errors=[];timer=time.monotonic()
    for b in range(20,40):
        actual=unbits(h[:,b,-1]);target=unbits(h[:,b+1,-1]);idx=params[f'B{b}_idx'];metrics={};se={}
        for m in METHODS:
            if m=='mean_next':one=np.broadcast_to(params[f'B{b}_mean'],target.shape);states[m]=one.copy()
            else:
                kind='direct' if m=='state_direct' else 'residual' if m=='state_residual' else 'remainder'
                p={k:params[f'B{b}_{kind}_{k}'] for k in ['center','scale','coef']};write=0
                if m in CONTEXT and b in [28,29]:
                    bi=[28,29].index(b);write=fore[CONTEXT[m]][:,bi]@fk[f'wo_{bi}'].T
                one=predict(actual[:,idx],p)+write
                update=predict(states[m][:,idx],p)+write
                if kind=='residual':one+=actual;states[m]+=update
                else:states[m]=update
            err=np.square(target-states[m]).sum(-1);se[m]=err
            metrics[m]={'teacher_forced_one_step_MSE':float(np.square(target-one).sum(-1).mean()),'recursive_MSE':float(err.mean()),
                'recursive_relative_error':float(np.sqrt(err.sum()/max(np.square(target).sum(),1e-12)))}
            assert np.isfinite(states[m]).all()
        steps.append({'block':b,'methods':metrics});errors.append(np.stack([se[m] for m in METHODS],axis=1));D.log(split+'/recurrence',b-19,20,timer)
    close(h);params.close();fore.close();fk.close();path=root/f'{split}-update-final.npz';assert not path.exists();np.savez(path,**states,per_row_errors=np.stack(errors,axis=1))
    D.save(f'{D.MODEL}/{split}-update.json',{'steps':steps,'fit_sha256':info['fit_sha256'],'final_sha256':D.sha(path),
        'scope':'Observed hs20 prefix once. PredictedV recursion has no true future Q/K/V or late final states as input. All later non-target computations are fitted approximations. TrueVpast variant remains privileged.'})

def output(split):
    import torch
    from source_relay2_behavior import Runtime
    D.verify_internal();D.admission(split);root=D.OUT/D.MODEL;rows=D.read(D.OUT/'qwen14-materials.json')['splits'][split]
    data=np.load(root/f'{split}-update-final.npz');h=load(split,'hidden');base=D.read(root/f'{split}-behavior.json')['rows'];saved={m:[] for m in METHODS};timer=time.monotonic()
    with D.gpu_lease(split+'/recurrence-output'),torch.inference_mode():
        rt=Runtime(D.MODEL)
        for lo in range(0,len(rows),128):
            hi=min(lo+128,len(rows));rs=rows[lo:hi];raw=torch.from_numpy(np.array(h[lo:hi,40,-1])).view(torch.bfloat16).cuda()
            _,lg=rt.finish(raw);original=rt.score(lg,rs)
            assert all(a['top1']==b['top1'] and a['scores']==b['scores'] for a,b in zip(original,base[lo:hi]))
            lp=lg.log_softmax(-1);prob=lp.exp();top=lg.argmax(-1)
            for m in METHODS:
                state=torch.tensor(data[m][lo:hi],device='cuda',dtype=torch.bfloat16);_,pred=rt.finish(state);kl=(prob*(lp-pred.log_softmax(-1))).sum(-1)
                items=rt.score(pred,rs);agree=pred.argmax(-1)==top
                for i,item in enumerate(items):item.update(KL=float(kl[i]),top1_matches_model=bool(agree[i]))
                saved[m]+=items;del state,pred
            del raw,lg,lp,prob,top;gc.collect();torch.cuda.empty_cache();D.log(split+'/recurrence-output',hi,len(rows),timer)
    close(h);data.close();summary={}
    for m,items in saved.items():
        summary[m]={'KL_mean':float(np.mean([x['KL'] for x in items])),'top1_agreement':float(np.mean([x['top1_matches_model'] for x in items])),
            'top1_accuracy':float(np.mean([x['top1_correct'] for x in items])),'forced_accuracy':float(np.mean([x['forced_correct'] for x in items]))}
    comparisons={}
    for other in ['state_direct','state_residual','ridge_predV_direct','aware_trueVpast_direct','meanV_direct']:
        gain=np.array([b['KL']-a['KL'] for a,b in zip(saved['aware_predV_direct'],saved[other])]);comparisons[other]=M.paired(rows,gain,split+'/recurrence/'+other)
    D.save(f'{D.MODEL}/{split}-update-output.json',{'summary':summary,'rows':saved,'aware_predV_KL_gain_comparisons':comparisons,
        'actual_final_logits_replay_exact':True,'scope':'Single next-token output distribution; autoregressive continuation fidelity not tested.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','evaluate','output']);ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='fit':fit()
    else:globals()[a.stage](a.split)
