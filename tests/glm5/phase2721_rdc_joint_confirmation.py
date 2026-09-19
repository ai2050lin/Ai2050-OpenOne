"""Untouched fresh material only after a verifiable frozen rule/selection manifest."""
import gc
from rdc_joint_common import *
from rdc_joint_capture import capture,ledger
from rdc_joint_features import build,load_features
from rdc_joint_kernels import CURRENT,TEMPORAL,all_dots,gram_for,error_report
from rdc_relation_estimators import load_model,predict
from phase2716_rdc_relation_probability import Readout


def frozen_check():
    frozen=read(BASE/'frozen.json')
    for path,digest in frozen['files'].items():assert sha(BASE/path)==digest,('Frozen file changed',path)
    return frozen


def evaluate():
    start=time.monotonic()
    frozen=frozen_check();out=BASE/'confirmation'
    if (out/'result.json').exists():return read(out/'result.json')
    train_raw,train_y,_,trainmeta=load_features()
    tr=np.array(read(BASE/'rules/indices.json')['train'])
    training={k:v[tr] for k,v in train_raw.items() if not k.endswith('query_proposal')}
    del train_raw
    raw,y,post,meta=load_features(True)
    dots,_=all_dots(raw,scales=read(BASE/'rules/scales.json'),other=training)
    npz(out/'fresh_vs_train_full_feature_grams.npz',**dots)
    del raw,training
    rd=Readout();result={};reports=[]
    try:
        fit=read(BASE/'rules/result.json')
        objective=read(BASE/'probability_training/protocol.json')['routes']
        for scope,names,a,b in [('current',CURRENT,0,5120),('temporal',TEMPORAL,5120,7680)]:
            target=y[:,a:b];subpost=post[:,int(scope=='temporal')]
            observed=[m['observed_current_output_token' if scope=='current' else 'observed_temporal_output_token'] for m in meta]
            baseline=None;rr={};preds={}
            for name in names:
                model=load_model(BASE/'rules'/scope/name)
                info=fit[scope]['candidates'][name]
                p=predict(model,gram_for(dots,scope,name,info['mix']))
                if baseline is None:baseline=p
                preds[name]=p
                rr[name]=error_report(p,target,meta,baseline)
                if scope=='current':rr[name]['by_target_layer']={k:error_report(p[:,u:v],target[:,u:v],meta,baseline[:,u:v]) for k,u,v in [('H23',0,2560),('H36',2560,5120)]}
                npz(out/'predictions'/f'{scope}_{name}.npz',prediction=p)
                reports.append(rd.evaluate(p[:,-2560:],subpost,observed,meta,out/'probability',name,scope,'fresh'))
                print('JOINT_CONFIRM_STATE',scope,name,rr[name]['anchor_MSE'],flush=True)
            for name in objective[scope]:
                model=load_model(BASE/'probability_training'/scope/name)
                info=fit[scope]['candidates'][name]
                p=predict(model,gram_for(dots,scope,name,info['mix']))
                label='KL_'+name
                rr[label]=error_report(p,target[:,-2560:],meta,preds[name][:,-2560:])
                npz(out/'predictions'/f'{scope}_{label}.npz',prediction=p)
                reports.append(rd.evaluate(p,subpost,observed,meta,out/'probability',label,scope,'fresh'))
            matched={};dfinfo=fit[scope]['matched_effective_df'];base_name=fit[scope]['matched_df_baseline']
            for name,info in dfinfo.items():
                with np.load(BASE/'rules'/scope/'matched_df'/f'{name}.npz') as z:model={k:z[k] for k in ('alpha','mean','target_scale')}
                p=predict(model,gram_for(dots,scope,name,info['mix']))
                matched[name]=error_report(p,target,meta,preds[base_name])
            reports.append(rd.evaluate(target[:,-2560:],subpost,observed,meta,out/'probability','actual_H36_FP32_floor',scope,'fresh'))
            result[scope]={'candidates':rr,'matched_df':matched}
    finally:rd.close()
    paired=[]
    for scope,base,candidate in [('current','current_linear','KL_current_linear'),('current','current_quadratic','KL_current_quadratic'),
                                 ('temporal','embedding_bilinear','KL_embedding_bilinear'),('temporal','history_trilinear','KL_history_trilinear'),
                                 ('temporal','KL_embedding_bilinear','KL_history_trilinear')]:
        aa=read(out/'probability'/f'fresh_{scope}_{base}.json')['rows'];bb=read(out/'probability'/f'fresh_{scope}_{candidate}.json')['rows']
        assert [(r['sample_id'],r['anchor']) for r in aa]==[(r['sample_id'],r['anchor']) for r in bb]
        paired.append({'scope':scope,'baseline':base,'candidate':candidate,
            'KL_baseline_minus_candidate':paired_summary([a['KL']-b['KL'] for a,b in zip(aa,bb)],[r['source_group'] for r in aa])})
    report={'timestamp':stamp(),'frozen_sha':sha(BASE/'frozen.json'),'sources':256,'anchors':512,
        'declared_groups':len({m['source_group'] for m in meta}),'choices_unchanged':frozen['choices'],'state':result,'probability':reports,'paired_KL':paired,
        'confirmation_not_fitted_or_selected':True,'scope':'Independent GUM official-test documents and Chinese translated PUD sentences; conditional one-step states/probability, not unrefreshed generation or semantic correctness.'}
    save(out/'result.json',report)
    ledger('joint_fresh_confirmation_evaluation',time.monotonic()-start,sources=256)
    guard()
    print('JOINT_CONFIRMATION_COMPLETE',report['choices_unchanged'],usage(),flush=True)
    return report


def main():
    frozen_check()
    if not (BASE/'features/fresh/complete.json').exists():
        cache=capture(fresh=True)
        try:build(cache)
        finally:cache.clear()
    evaluate()


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
