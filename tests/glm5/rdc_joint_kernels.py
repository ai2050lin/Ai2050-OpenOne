"""Full-block positive-semidefinite kernels, explicit information and capacity controls."""
from rdc_joint_common import *
from rdc_relation_estimators import Bank,select,predict,load_model
from rdc_joint_prior_rules import array_hashes
from rdc_joint_features import load_features
from rdc_joint_capture import ledger

CURRENT = ('current_linear','current_quadratic','raw_mean','rms_mean','direction_scale','rms_relation',
           'rms_relation_bilinear','native_context','native_context_bilinear','native_values_permuted')
TEMPORAL = ('state_only','embedding_linear','embedding_bilinear','history_context','history_trilinear',
            'history_values_permuted','actual_query_extra_information')
DIAGNOSTIC = {'actual_query_extra_information'}


def route_dots(d,scope,name):
    if scope=='current':
        x = d['current_current']
        if name=='current_linear':return {'current':x},'current'
        if name=='current_quadratic':return {'current':x},'full_quadratic'
        key = {'rms_relation_bilinear':'rms_relation','native_context_bilinear':'native_context',
               'native_values_permuted':'native_permuted'}.get(name,name)
        r = d['current_'+key]
        if name.endswith('_bilinear'):r = (r+x*r)*.5
        return {'current':x,'history_mean':r},'history_mean'
    h,e = d['temporal_previous'],d['temporal_embedding']
    if name=='state_only':return {'current':h},'current'
    if name=='embedding_linear':return {'current':h,'history_mean':e},'history_mean'
    if name=='embedding_bilinear':return {'current':h,'history_mean':(e+h*e)*.5},'history_mean'
    # Same fixed complete H/E pair base for all history-context branches, mixture includes zero.
    base = (h+e+h*e)/3
    key = {'history_values_permuted':'context_permuted',
           'actual_query_extra_information':'context_actual_query_extra_information'}.get(name,'context')
    c = d['temporal_'+key]
    history = (c+h*c+e*c+h*e*c)/4 if name=='history_trilinear' else c
    return {'current':base,'history_mean':history},'history_mean'


def all_dots(raw,train=None,scales=None,other=None):
    active = {k:v for k,v in raw.items() if not k.endswith('query_proposal')}
    if scales is None:
        scales = {k:max(float(np.mean(np.sum(v[train].reshape(len(train),-1).astype(float)**2,axis=1))),1e-15) for k,v in active.items()}
    other = active if other is None else other
    dots = {}
    for k,v in active.items():
        a = v.reshape(len(v),-1)
        b = other[k].reshape(len(other[k]),-1)
        # Block columns only as an exact sum, never rank reduction.
        answer = np.zeros((len(a),len(b)),float)
        for i in range(0,a.shape[1],8192):answer += a[:,i:i+8192].astype(float)@b[:,i:i+8192].astype(float).T
        dots[k] = answer/scales[k]
    return dots,scales


def gram_for(dots,scope,name,mix):
    d,k = route_dots(dots,scope,name)
    return Bank.gram(d,k,mix)


def error_report(p,y,meta,baseline=None):
    mse = np.mean((p.astype(float)-y)**2,axis=1)
    group = [m['source_group'] for m in meta]
    result = {'anchor_MSE':float(mse.mean()),'group_MSE':paired_summary(mse,group),
        'by_language':{l:paired_summary(mse[[m['language']==l for m in meta]], [m['source_group'] for m in meta if m['language']==l]) for l in ('en','zh')},
        'by_family':{f:paired_summary(mse[[f in m['language_mode_families'] for m in meta]], [m['source_group'] for m in meta if f in m['language_mode_families']]) for f in sorted({f for m in meta for f in m['language_mode_families']})}}
    if baseline is not None:
        delta = np.mean((baseline.astype(float)-y)**2,axis=1)-mse
        result['baseline_minus_candidate'] = paired_summary(delta,group)
    return result


def fit_main():
    out = BASE/'rules'
    if (out/'result.json').exists():return read(out/'result.json')
    start = time.monotonic()
    snapshot(Path(__file__))
    raw,y,post,meta = load_features()
    train,val,test = [np.array([i for i,m in enumerate(meta) if m['split']==s]) for s in ('train','validation','test')]
    dots,scales = all_dots(raw,train)
    npz(out/'all_full_feature_grams.npz',**dots)
    save(out/'scales.json',scales)
    save(out/'indices.json',{'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()})
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'current':CURRENT,'temporal':TEMPORAL,
        'training_sources':320,'validation_sources':96,'test_sources':96,'anchors_each':2,
        'objective':'Training target centered, each full H block scaled by training RMS; joint current H23/H36 validation selects mix/ridge. Temporal target only next H36.',
        'grid':{'mix':[0,.1,.25,.5,1],'ridge':[.00001,.001,.1,1,10]},
        'capacity':'Full native coordinates; dot product Gram uses all axes. Quadratic/bilinear/trilinear are exact polynomial feature expansions, not native gear factorizations. Record effective trace df and match df to linear current or embedding-bilinear temporal baseline.',
        'information':'Current and temporal feature scopes in features/main/complete.json; actual_query_extra_information excluded from primary choice.',
        'training_weighting':'Equal anchors (two per window). Uncertainty and reported group means cluster by declared document/content group. Not weighted document training or complete unseen-paraphrase split.',
        'simple_controls':['training_mean','common_increment_from_current_or_previous'],
        'test_rule':'No test/confirmation outcomes select fits; plots/family splits are retrospective evaluation.'})
    allresult = {}
    for scope,names,a,b,blocks in [('current',CURRENT,0,5120,[(0,2560),(2560,5120)]),('temporal',TEMPORAL,5120,7680,[(0,2560)])]:
        target = y[:,a:b]
        result,preds = {},{}
        baseline = None
        for name in names:
            folder = out/scope/name
            dd,kk = route_dots(dots,scope,name)
            model,info,grid = select(dd,kk,target,train,val,blocks)
            p = predict(model,Bank.gram(dd,kk,info['mix'])[:,train])
            if baseline is None:baseline = p
            preds[name] = p
            report = {'route':name,'scope':scope,**info,'grid':grid,'coefficient_sha':array_hashes(model),
                'extra_information':name in DIAGNOSTIC,'coefficient_elements':int(model['alpha'].size),
                'validation':error_report(p[val],target[val],[meta[i] for i in val]),
                'test':error_report(p[test],target[test],[meta[i] for i in test],baseline[test])}
            if scope=='current':
                report['test_by_target_layer'] = {k:error_report(p[test,u:v],target[test,u:v],[meta[i] for i in test],baseline[test,u:v]) for k,u,v in [('H23',0,2560),('H36',2560,5120)]}
            npz(folder/'model.npz',**model)
            npz(folder/'predictions.npz',validation=p[val],test=p[test])
            npz(folder/'all_coordinate_test_MSE.npz',value=np.mean((p[test].astype(float)-target[test])**2,0).astype(np.float32))
            save(folder/'result.json',report)
            result[name] = report
            print('JOINT_FIT',scope,name,'mix',info['mix'],'df',round(info['effective_df'],2),'testMSE',report['test']['anchor_MSE'],flush=True)
        winner = min([n for n in names if n not in DIAGNOSTIC],key=lambda n:(result[n]['validation_normalized_MSE'],names.index(n)))
        base_name = 'current_linear' if scope=='current' else 'embedding_bilinear'
        df = result[base_name]['effective_df']
        matched = {}
        for name in names:
            if name in DIAGNOSTIC or name==base_name:continue
            dd,kk = route_dots(dots,scope,name)
            model,info,_ = select(dd,kk,target,train,val,blocks,fixed_df=df,fixed_mix=result[name]['mix'])
            p = predict(model,Bank.gram(dd,kk,info['mix'])[:,train])
            npz(out/scope/'matched_df'/f'{name}.npz',**model,validation=p[val],test=p[test])
            matched[name] = {**info,'test':error_report(p[test],target[test],[meta[i] for i in test],preds[base_name][test])}
        raw_previous = raw['current_current'] if scope=='current' else raw['temporal_previous']
        raw_previous = np.tile(raw_previous,(1,2)) if scope=='current' else raw_previous
        control = {'training_mean':np.broadcast_to(target[train].mean(0),target.shape),
                   'common_increment':raw_previous+(target[train]-raw_previous[train]).mean(0)}
        controls = {k:error_report(p[test],target[test],[meta[i] for i in test]) for k,p in control.items()}
        allresult[scope] = {'candidates':result,'validation_MSE_winner':winner,'matched_df_baseline':base_name,
                            'matched_effective_df':matched,'simple_controls':controls}
    report = {'timestamp':stamp(),'current':allresult['current'],'temporal':allresult['temporal'],
              'full_native_coordinate_policy':True,'fresh_responses_seen':False,'source':snapshot(Path(__file__))}
    save(out/'result.json',report)
    ledger('joint_MSE_kernel_fit',time.monotonic()-start,train_anchors=len(train))
    guard()
    print('JOINT_ALL_MSE_FITS', {k:v['validation_MSE_winner'] for k,v in allresult.items()},usage(),flush=True)
    return report


class Rule:
    def __init__(self,scope,name,*,kl=False):
        self.scope,self.name,self.kl = scope,name,kl
        self.info = read(BASE/'rules'/scope/name/'result.json')
        folder = BASE/('probability_training' if kl else 'rules')/scope/name
        self.model = load_model(folder)
        raw,y,post,meta = load_features()
        tr = read(BASE/'rules/indices.json')['train']
        self.train = {k:v[tr] for k,v in raw.items() if not k.endswith('query_proposal') and k.startswith(scope+'_')}
        self.scales = read(BASE/'rules/scales.json')
    def __call__(self,features):
        raw = {self.scope+'_'+k:np.atleast_2d(v) for k,v in features.items() if self.scope+'_'+k in self.train}
        # Single-item matrix-valued relation messages need an explicit leading sample dimension.
        for k in raw:
            if raw[k].ndim==self.train[k].ndim-1:raw[k] = raw[k][None]
        dots,_ = all_dots(raw,scales=self.scales,other=self.train)
        return predict(self.model,gram_for(dots,self.scope,self.name,self.info['mix']))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):fit_main()
