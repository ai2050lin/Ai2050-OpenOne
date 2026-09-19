"""Actual full-vocabulary KL optimization, then validation-only choices and fresh freeze."""
import gc
from rdc_joint_common import *
from rdc_joint_features import load_features
from rdc_joint_kernels import CURRENT,TEMPORAL,DIAGNOSTIC,gram_for,error_report
from rdc_relation_estimators import load_model,predict
from rdc_joint_prior_rules import array_hashes
from rdc_joint_capture import ledger
from phase2716_rdc_relation_probability import Readout


def gradient_logprob(rd,h):
    t=rd.torch
    return ((h*t.rsqrt(h.square().mean(-1,keepdim=True)+rd.eps)*rd.gamma)@rd.w.T).log_softmax(-1)


def optimize(rd,scope,name,gram,model,target,post,train,val):
    t=rd.torch
    out=BASE/'probability_training'/scope/name
    if (out/'result.json').exists():return read(out/'result.json')
    # Only H36 is optimized; joint current H23 coefficients remain separately fixed in MSE model.
    offset = 2560 if scope=='current' else 0
    initial={k:v[:,offset:].copy() if k=='alpha' else v[offset:].copy() for k,v in model.items()}
    ktrain=t.as_tensor(gram[np.ix_(train,train)],dtype=t.float32,device='cuda')
    kval=t.as_tensor(gram[np.ix_(val,train)],dtype=t.float32,device='cuda')
    mean=t.as_tensor(initial['mean'],device='cuda'); scale=t.as_tensor(initial['target_scale'],device='cuda')
    posttrain=t.as_tensor(post[train],dtype=t.bfloat16,device='cuda')
    postval=t.as_tensor(post[val],dtype=t.bfloat16,device='cuda')
    basealpha=t.as_tensor(initial['alpha'],device='cuda')
    def evaluate(alpha):
        values=[]
        with t.inference_mode():
            for i in range(0,len(val),16):
                h=(kval[i:i+16]@alpha)*scale+mean
                lp=gradient_logprob(rd,h)
                lq=(postval[i:i+16]@rd.wb.T).float().log_softmax(-1)
                values.extend((lq.exp()*(lq-lp)).sum(-1).cpu().tolist())
        return float(np.mean(values))
    initial_kl=evaluate(basealpha)
    best_value=initial_kl;best=initial['alpha'].copy();choice={'learning_rate':0,'epoch':0,'validation_KL':initial_kl}
    curve=[]
    for lr in (1e-4,3e-4):
        alpha=t.nn.Parameter(basealpha.clone())
        opt=t.optim.Adam([alpha],lr=lr)
        generator=t.Generator(device='cpu').manual_seed(2720)
        for epoch in range(1,9):
            order=t.randperm(len(train),generator=generator).tolist();losses=[]
            for begin in range(0,len(order),32):
                ix=order[begin:begin+32]
                opt.zero_grad(set_to_none=True)
                h=(ktrain[ix]@alpha)*scale+mean
                lp=gradient_logprob(rd,h)
                with t.no_grad():lq=(posttrain[ix]@rd.wb.T).float().log_softmax(-1)
                loss=(lq.exp()*(lq-lp)).sum(-1).mean()
                assert t.isfinite(loss)
                loss.backward()
                assert t.isfinite(alpha.grad).all()
                opt.step();losses.append(float(loss.detach()))
            value=evaluate(alpha)
            item={'learning_rate':lr,'epoch':epoch,'training_batch_mean_KL':float(np.mean(losses)),'validation_KL':value}
            curve.append(item)
            if value<best_value:
                best_value=value;best=alpha.detach().cpu().numpy().copy();choice=item
            print('JOINT_KL_TRAIN',scope,name,'lr',lr,'epoch',epoch,'validation',round(value,6),flush=True)
        del alpha,opt
    optimized={**initial,'alpha':best}
    npz(out/'model.npz',**optimized)
    prediction=predict(optimized,gram[:,train])
    indices=read(BASE/'rules/indices.json');test=np.array(indices['test'])
    npz(out/'predictions.npz',validation=prediction[val],test=prediction[test])
    report={'timestamp':stamp(),'scope':scope,'route':name,'initial_validation_KL_FP32_kernel':initial_kl,'selected':choice,
        'curve':curve,'coefficient_sha':array_hashes(optimized),'coefficient_elements':int(best.size),
        'objective':'Mean exact151936-way native-to-predicted KL, no truncation/no sampled softmax/no MSE penalty.',
        'information_and_capacity':'Same frozen kernel and training reference points as MSE route, same H36 coefficient count. Mean/target-scale/gamma/unembedding fixed. KL-optimized statistical effective df is not claimed equal to ridge trace df.',
        'optimization':'Adam, fixed seed2720, batch32, two predeclared rates,8epochs; initial epoch0 eligible. Float32 complete norm/head; TF32 off. Finite gradients checked.',
        'no_test_or_confirmation_selection':True}
    save(out/'result.json',report)
    del ktrain,kval,mean,scale,posttrain,postval,basealpha
    gc.collect();t.cuda.empty_cache()
    return report


def main():
    import torch
    torch.set_num_threads(4)
    start=time.monotonic()
    fit=read(BASE/'rules/result.json')
    out=BASE/'probability_training'
    current_winner=fit['current']['validation_MSE_winner'];temporal_winner=fit['temporal']['validation_MSE_winner']
    routes={'current':list(dict.fromkeys(['current_linear',current_winner,'native_context_bilinear'])),
            'temporal':list(dict.fromkeys([temporal_winner,'history_trilinear']))}
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'routes':routes,
        'scope':'MSE validation winners plus fixed plain-current and maximal selected-native-history challengers; objective-comparison design finalized after main MSE observations, before any fresh response. This is not retrospectively called pre-main preregistration.',
        'objective':'Optimize actual full-vocabulary KL coefficients, not merely choose an MSE fit by validation KL.',
        'learning_rates':[1e-4,3e-4],'epochs':8,'batch':32,'seed':2720,'initial_epoch_zero_eligible':True,
        'reference':'Original actual BF16 postnorm and head; batch32 training and batch16 validation/evaluation. Different-shape floor measured separately.',
        'fresh_sources':256,'fresh_responses_seen':False})
    raw,y,post,meta=load_features()
    del raw
    with np.load(BASE/'rules/all_full_feature_grams.npz') as z:dots={k:z[k] for k in z.files}
    indices=read(BASE/'rules/indices.json');train,val,test=[np.array(indices[s]) for s in ('train','validation','test')]
    rd=Readout();trainreports=[];reports=[];state=[]
    try:
        for scope,names in routes.items():
            for name in names:
                model=load_model(BASE/'rules'/scope/name);info=fit[scope]['candidates'][name]
                gram=gram_for(dots,scope,name,info['mix'])
                actual=y[:,2560:5120] if scope=='current' else y[:,5120:7680]
                refpost=post[:,0 if scope=='current' else 1]
                trainreports.append(optimize(rd,scope,name,gram,model,actual,refpost,train,val))
        for scope,names in [('current',CURRENT),('temporal',TEMPORAL)]:
            actual=y[:,2560:5120] if scope=='current' else y[:,5120:7680]
            refpost=post[:,0 if scope=='current' else 1]
            for split,ii in [('validation',val),('test',test)]:
                submeta=[meta[i] for i in ii]
                observed=[m['observed_current_output_token' if scope=='current' else 'observed_temporal_output_token'] for m in submeta]
                candidates=[]
                for name in names:
                    with np.load(BASE/'rules'/scope/name/'predictions.npz') as z:p=z[split][:,-2560:]
                    candidates.append((name,p,False))
                for name in routes[scope]:
                    with np.load(out/scope/name/'predictions.npz') as z:p=z[split]
                    candidates.append(('KL_'+name,p,False))
                candidates.extend([('actual_H36_FP32_floor',actual[ii],True),('training_mean',np.broadcast_to(actual[train].mean(0),(len(ii),2560)),True)])
                for name,p,diagnostic in candidates:
                    cp=BASE/'probability'/f'{split}_{scope}_{name}.json'
                    r=read(cp)['summary'] if cp.exists() else rd.evaluate(p,refpost[ii],observed,submeta,BASE/'probability',name,scope,split)
                    reports.append({**r,'extra_information':name in DIAGNOSTIC,'arithmetic_or_simple_control':diagnostic})
                    state.append({'split':split,'scope':scope,'route':name,'state':error_report(p,actual[ii],submeta)})
    finally:rd.close()
    choices={}
    for scope in ('current','temporal'):
        eligible=[r for r in reports if r['scope']==scope and r['split']=='validation' and not r['extra_information'] and not r['arithmetic_or_simple_control']]
        winner=min(eligible,key=lambda r:(r['KL'],r['route']))['route']
        choices[scope+'_MSE']=fit[scope]['validation_MSE_winner']
        choices[scope+'_KL']=winner
        # Force a genuine history-state candidate for the richer autonomous-state comparison, regardless of winning.
        if scope=='temporal':
            history=[r for r in eligible if r['route'] in ('history_context','history_trilinear','KL_history_trilinear')]
            choices['temporal_history_KL']=min(history,key=lambda r:(r['KL'],r['route']))['route']
    report={'timestamp':stamp(),'optimization':trainreports,'probability':reports,'state':state,'choices':choices,
        'full_vocabulary':rd.vocab,'fresh_responses_seen':False,'source':snapshot(Path(__file__))}
    save(out/'result.json',report)
    files={}
    for folder in ('rules','features/main','probability_training'):
        for p in (BASE/folder).rglob('*'):
            if p.is_file():files[str(p.relative_to(BASE))]=sha(p)
    for p in (material_path(),material_path(True),BASE/'main/archive_manifest.json',BASE/'query_proposal/frozen.json'):
        files[str(p.relative_to(BASE))]=sha(p)
    immutable(BASE/'frozen.json',{'timestamp':stamp(),'choices':choices,'files':files,
        'sources':{name:snapshot(ROOT/'tests/glm5'/name) for name in ('rdc_joint_features.py','rdc_joint_native_attention.py','rdc_joint_kernels.py','phase2720_rdc_probability_training.py')},
        'fresh_sources':256,'fresh_responses_seen':False,'selection':'Validation only; MSE or full-vocabulary KL objectives distinctly identified. Test and forthcoming independent confirmation are evaluations.'})
    ledger('joint_full_vocabulary_probability_training',time.monotonic()-start)
    guard()
    print('JOINT_FROZEN',choices,usage(),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
