"""Balanced-family source-cluster comparisons; no post-test route reselection."""
from collections import defaultdict
from rdc_construction_common import *
from phase2746_rdc_history_prediction_contract import OUT


def summarize(records,names,metrics,seed):
    groups=defaultdict(lambda:defaultdict(list))
    for r in records:groups[(r['family'],r['source_group'])][r['route']].append(r)
    keys=sorted(groups);families=sorted({k[0] for k in keys});matrix=np.empty((len(keys),len(names),len(metrics)))
    for g,key in enumerate(keys):
        for j,name in enumerate(names):
            rr=groups[key][name];assert rr,(key,name)
            matrix[g,j]=[np.mean([float(r[m]) for r in rr]) for m in metrics]
    mean_weights=np.zeros(len(keys));bootstrap=np.zeros((2000,len(keys)));rng=np.random.default_rng(seed)
    for family in families:
        ii=[i for i,k in enumerate(keys) if k[0]==family];n=len(ii)
        mean_weights[ii]=1/(len(families)*n)
        bootstrap[:,ii]=rng.multinomial(n,np.full(n,1/n),size=2000)/(len(families)*n)
    actual=np.einsum('g,grm->rm',mean_weights,matrix)
    draws=np.einsum('bg,grm->brm',bootstrap,matrix,optimize=True)
    intervals=np.quantile(draws,[.025,.975],axis=0)
    direct=next(i for i,n in enumerate(names) if n=='native_history__true_correspondence__direct_complete_coordinate_H36')
    token=names.index('training_current_input_token_H36_mean')
    result=[]
    for r,name in enumerate(names):
        values={}
        for m,metric in enumerate(metrics):
            values[metric]={'mean':float(actual[r,m]),'interval95':intervals[:,r,m].tolist(),
                'minus_direct_mean':float(actual[r,m]-actual[direct,m]),
                'minus_direct_interval95':np.quantile(draws[:,r,m]-draws[:,direct,m],[.025,.975]).tolist(),
                'minus_current_token_baseline_mean':float(actual[r,m]-actual[token,m]),
                'minus_current_token_baseline_interval95':np.quantile(draws[:,r,m]-draws[:,token,m],[.025,.975]).tolist()}
        result.append({'route':name,'metrics':values})
    return {'source_groups':len(keys),'families':families,'weighting':'Equal families, equal source_group within family, equal observations within source; paired stratified2000source bootstrap.',
        'scope':'Conditional on this fixed exposed-material split and frozen fit; coordinates, routes, worlds and steps are not independent replicates.',
        'results':result}


def main():
    start=time.monotonic();directory=OUT/'analysis';meta=read(OUT/'test/heldout.json');assert meta['all_passed']
    records=gzread(OUT/'test/records.json.gz');names=meta['routes'];reports=[]
    metrics=['postnorm_MSE','KL_reference_prediction','TV','argmax_agrees_original_native']
    families=sorted({r['family'] for r in records})
    subsets=[('all',lambda r:True),('natural',lambda r:r['kind']=='natural'),('controlled',lambda r:r['kind']=='controlled')]
    subsets += [(f,lambda r,f=f:r['family']==f) for f in families]
    for subset,fn in subsets:
        selected=[r for r in records if fn(r)]
        for horizon in ['first_position','first3_native_history_positions']:
            rr=[r for r in selected if horizon!='first_position' or r['step']==0]
            value=summarize(rr,names,metrics,2746)
            value.update(subset=subset,horizon=horizon,points_per_route=len(rr)//len(names));reports.append(value)
    pairs=gzread(OUT/'test/pair_changes.json.gz')
    for r in pairs:r['pair_MSE_minus_zero']=r['pair_change_MSE']-r['zero_change_MSE']
    pair_reports=[]
    for family in ['all',*sorted({r['family'] for r in pairs})]:
        rr=[r for r in pairs if family=='all' or r['family']==family]
        value=summarize(rr,names,['pair_change_MSE','pair_MSE_minus_zero','answer_aligned_separation'],2747)
        value.update(subset=family,pairs_per_route=len(rr)//len(names));pair_reports.append(value)
    spec=read(OUT/'test/protocol.json');answer_ids=spec['answer_first_token_ids']
    answers=[r.copy() for r in records if r['kind']=='controlled' and r['step']==0]
    for r in answers:
        truth=r['target'] in ['Yes','是'];gold=answer_ids[r['language']][0 if truth else 1]
        r['first_token_gold_match']=r['predicted_argmax']==gold
        r['candidate_conditional_gold_match']=(r['conditional_yes_probability']>=.5)==truth
    answer_reports=summarize(answers,names,['first_token_gold_match','candidate_conditional_gold_match'],2748)
    answer_reports['scope']='First actual next-token or conditional two-candidate judgment on fixed native histories; not full answers, stopping or self-fed behavior.'
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'test_result_sha256':sha(OUT/'test/heldout.json'),
        'frozen_sha256':sha(OUT/'frozen.json'),'reports':reports,'pair_reports':pair_reports,'controlled_first_answers':answer_reports,
        'no_post_test_selection':True,'independent_confirmation_executed':False,'autonomous_deployment_executed':False,
        'scientific_boundary':'Separate full-state fidelity, all-vocabulary fidelity, matched-pair relation change and language answer. Extra pastKV is genuinely available before the current token; current future states are not. A local conditional decoder is not yet an all-future sufficient state or language mechanism.',
        'seconds':time.monotonic()-start}
    save(directory/'result.json',result);ledger('phase2746_history_analysis',result['seconds'])
    print('HISTORY_ANALYSIS_COMPLETE',len(reports),len(pair_reports),result['seconds'],flush=True)


if __name__=='__main__':main()
