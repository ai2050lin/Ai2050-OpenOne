"""Full self-fed outcomes, source-group intervals and complete-coordinate drift."""
from collections import defaultdict
from itertools import groupby
from rdc_construction_common import *
from phase2746_rdc_confirmation_material import OUT
from phase2746_rdc_history_prediction_contract import OUT as PRED

ROUTES=['native_B1_cache','frozen_direct','frozen_generalQ_native']


def grouped_summary(records,metrics,seed):
    groups=defaultdict(dict)
    for r in records:groups[(r['family'],r['source_group'])].setdefault(r['route'],[]).append(r)
    keys=sorted(groups);families=sorted({k[0] for k in keys});values=np.array([[[np.mean([r[m] for r in groups[g][route]]) for m in metrics] for route in ROUTES] for g in keys])
    w=np.zeros(len(keys));bw=np.zeros((2000,len(keys)));rng=np.random.default_rng(seed)
    for family in families:
        ix=[i for i,g in enumerate(keys) if g[0]==family];n=len(ix);w[ix]=1/(len(families)*n)
        bw[:,ix]=rng.multinomial(n,np.full(n,1/n),size=2000)/(len(families)*n)
    actual=np.einsum('g,grm->rm',w,values);boot=np.einsum('bg,grm->brm',bw,values,optimize=True)
    return {'source_groups':len(keys),'families':families,'weighting':'Equal families and equal source groups within family;2000source bootstrap, worlds/languages are not independent replicates.',
        'routes':[{'route':route,'metrics':{m:{'mean':float(actual[r,j]),'interval95':np.quantile(boot[:,r,j],[.025,.975]).tolist(),
            'minus_native_mean':float(actual[r,j]-actual[0,j]),'minus_native_interval95':np.quantile(boot[:,r,j]-boot[:,0,j],[.025,.975]).tolist()}
            for j,m in enumerate(metrics)}} for r,route in enumerate(ROUTES)]}


def main():
    start=time.monotonic();directory=OUT/'autonomous';meta=read(directory/'result.json');assert meta['all_passed']
    if (directory/'analysis.json').exists():
        previous=read(directory/'analysis.json')
        if previous['source']['sha256']!=sha(__file__):
            revision=directory/'analysis_revisions'/previous['source']['sha256'][:16]
            revision.mkdir(parents=True,exist_ok=True)
            for name in ['analysis.json','metrics.json.gz','pair_metrics.json.gz','coordinate_trajectories.json.gz','examples.json.gz']:
                old=directory/name;copy=revision/name
                if not copy.exists():shutil.copyfile(old,copy)
                assert sha(old)==sha(copy)
            save(revision/'correction.json',{'timestamp':stamp(),'reason':'Correct unused pair metric label to actual opposite first tokens; use the actual strict_answer_only scorer key, add first-token B1/B8 counts. Main trajectories and scientific summary metrics not rerun or selected.',
                'new_source':snapshot(__file__),'previous_source':previous['source']})
    rows=gzread(OUT/'material.json.gz');material={r['sample_id']:r for r in rows};records=gzread(directory/'records.json.gz')
    native={r['sample_id']:r for r in records if r['route']==ROUTES[0]};b8={r['sample_id']:r for r in gzread(OUT/'native/records.json.gz')}
    metrics=[];trajectories=[];pairs=[];examples=[]
    cm=read(PRED/'deployment/coefficients.json')
    with np.load(BASE/cm['field_path']) as z:mean=z['feature_mean_FP64'];scale=z['feature_scale_FP64']
    fit=read(PRED/'fit/native_history.json')
    with np.load(BASE/fit['field_path']) as z:
        tr=np.sqrt(np.mean(z['train_Z'].astype(float)**2,-1));train_range={'minimum':float(tr.min()),'median':float(np.median(tr)),'q95':float(np.quantile(tr,.95)),'maximum':float(tr.max())}
    for index,r in enumerate(records):
        row=material[r['sample_id']];ids=r['generated_ids'];reference=native[r['sample_id']]['generated_ids'];common=0
        for a,b in zip(ids,reference):
            if a!=b:break
            common+=1
        maximum_run=max(len(list(g)) for _,g in groupby(ids));unique_bigrams=len(set(zip(ids,ids[1:])))
        rec={k:r[k] for k in ['sample_id','source_group','family','kind','language','route']}
        rec.update(generated_steps=len(ids),first_emitted_token_id=ids[0],EOS=r['EOS'],censored=r['censored'],same_first_token=ids[0]==reference[0],
            full_sequence_equal_native=ids==reference,common_prefix_tokens=common,first_divergence_position=None if ids==reference else common,
            maximum_identical_token_run=maximum_run,contains_run_at_least8=maximum_run>=8,
            distinct_bigram_fraction=unique_bigrams/max(len(ids)-1,1),
            native_B1_vs_B8_full_sequence_equal=reference==b8[r['sample_id']]['generated_ids'])
        if row['kind']=='controlled':
            rec.update(pair_id=row['pair_id'],world=row['world'],target=row['target'],
                correct_and_stopped=r['answer_scoring']['parsed_and_stopped_correct'],strict_answer_only=r['answer_scoring']['strict_answer_only'],
                first_token_gold_match=ids[0]==read(PRED/'test/protocol.json')['answer_first_token_ids'][row['language']][0 if row['truth'] else 1])
        metrics.append(rec)
        if r['kind']=='natural':
            assert sha(BASE/r['field_path'])==r['field_sha256']
            with np.load(BASE/r['field_path']) as z:
                if r['route']==ROUTES[0]:
                    h=unbits(z['hidden']).astype(float);post=unbits(z['postnorm']).astype(float)
                    hh=h[:,35];h12=h[:,12];xs=None;kv=None;q=None
                else:
                    h12=unbits(z['early_H0_H12'][:,12]).astype(float);h0=unbits(z['early_H0_H12'][:,0]).astype(float)
                    y=z['predicted_H35_H36_Q35'].astype(float);hh=y[:,:2560];post=z['compiled_postnorm'].astype(float);q=y[:,5120:]
                    xs=np.sqrt(np.mean(((np.concatenate([h0,h12,z['available_candidate']],-1)-mean)/scale)**2,-1))
                    kv=z['appended_V35'].astype(float)
                for step in range(len(ids)):
                    trace={k:rec[k] for k in ['sample_id','source_group','family','language','route']}
                    trace.update(step=step,current_emitted_token_id=ids[step],H12_RMS=float(np.sqrt(np.mean(h12[step]**2))),
                        H35_RMS=float(np.sqrt(np.mean(hh[step]**2))),postnorm_RMS=float(np.sqrt(np.mean(post[step]**2))),
                        same_generated_history_before_step=step<=common,
                        H35_kind='native' if r['route']==ROUTES[0] else 'prediction')
                    if xs is not None:trace.update(standardized_available_X_RMS=float(xs[step]),
                        above_training_max_X_RMS=bool(xs[step]>train_range['maximum']),current_V35_RMS=float(np.sqrt(np.mean(kv[step]**2))),
                        predicted_Q35_RMS=float(np.sqrt(np.mean(q[step]**2))))
                    trajectories.append(trace)
        if index%96==0:print('SELF_ANALYSIS',index,len(records),round(time.monotonic()-start,1),flush=True)
    for pair in sorted({r['pair_id'] for r in metrics if r['kind']=='controlled'}):
        for route in ROUTES:
            rr=[r for r in metrics if r.get('pair_id')==pair and r['route']==route];assert len(rr)==2
            pairs.append({'pair_id':pair,'source_group':rr[0]['source_group'],'family':rr[0]['family'],'route':route,
                'both_worlds_correct_and_stopped':all(r['correct_and_stopped'] for r in rr),'opposite_first_tokens':rr[0]['first_emitted_token_id']!=rr[1]['first_emitted_token_id']})
    reports=[]
    commonmetrics=['generated_steps','EOS','censored','same_first_token','full_sequence_equal_native','common_prefix_tokens',
        'maximum_identical_token_run','contains_run_at_least8','distinct_bigram_fraction']
    for subset in ['natural','controlled',*sorted({r['family'] for r in metrics})]:
        rr=[r for r in metrics if r['kind']==subset or r['family']==subset]
        names=commonmetrics+(['correct_and_stopped','first_token_gold_match'] if rr[0]['kind']=='controlled' else [])
        report=grouped_summary(rr,names,2756);report.update(subset=subset,expressions_per_route=len(rr)//3);reports.append(report)
    pair_reports=[]
    for family in ['all',*sorted({r['family'] for r in pairs})]:
        rr=[r for r in pairs if family=='all' or r['family']==family];report=grouped_summary(rr,['both_worlds_correct_and_stopped'],2757)
        report['subset']=family;pair_reports.append(report)
    trace_reports=[]
    for family in sorted({r['family'] for r in trajectories}):
        for step in range(32):
            rr=[r for r in trajectories if r['family']==family and r['step']==step]
            report=grouped_summary(rr,['H12_RMS','H35_RMS','postnorm_RMS','same_generated_history_before_step'],2758)
            report.update(family=family,step=step);trace_reports.append(report)
    # First hash-frozen source of each natural corpus, not chosen by goodness or badness.
    for family in ['natural_ewt','natural_cmrc']:
        row=next(r for r in rows if r['family']==family)
        examples.append({'material':row,'records':[r for r in records if r['sample_id']==row['sample_id']]})
    # Original frozen detailed tables stay intact; corrected labels/keys use
    # new explicit table identities. The current summary points to them.
    compressed(directory/'metrics_corrected.json.gz',metrics);compressed(directory/'coordinate_trajectories.json.gz',trajectories)
    compressed(directory/'pair_metrics_corrected.json.gz',pairs);compressed(directory/'examples.json.gz',examples)
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'input_sha256':sha(directory/'result.json'),
        'reports':reports,'pair_reports':pair_reports,'complete_coordinate_drift':trace_reports,'train_X_RMS_reference':train_range,
        'natural_B1_B8_full_sequence_disagreements':sum(native[r['sample_id']]['generated_ids']!=b8[r['sample_id']]['generated_ids'] for r in rows if r['kind']=='natural'),
        'controlled_B1_B8_full_sequence_disagreements':sum(native[r['sample_id']]['generated_ids']!=b8[r['sample_id']]['generated_ids'] for r in rows if r['kind']=='controlled'),
        'natural_B1_B8_first_token_disagreements':sum(native[r['sample_id']]['generated_ids'][0]!=b8[r['sample_id']]['generated_ids'][0] for r in rows if r['kind']=='natural'),
        'controlled_B1_B8_first_token_disagreements':sum(native[r['sample_id']]['generated_ids'][0]!=b8[r['sample_id']]['generated_ids'][0] for r in rows if r['kind']=='controlled'),
        'natural_own_X_beyond_training_range':[{ 'route':route,'count':sum(r.get('above_training_max_X_RMS',False) for r in trajectories if r['route']==route),
            'points':sum(r['route']==route for r in trajectories)} for route in ROUTES[1:]],
        'scope':'All512 frozen sources and three self-fed branches. Repetition and token divergence are observable behavior, not a complete natural-language quality metric. Different-history state drift is not a same-input forecast error. Case/source bootstrap covers these materials, not a universal law.',
        'autonomous_deployment_executed':True,'cold_start_prefix_extraction':False,'teacher_refresh_after_initialization':0,
        'detailed_metrics_file':'metrics_corrected.json.gz','pair_metrics_file':'pair_metrics_corrected.json.gz',
        'seconds':time.monotonic()-start}
    save(directory/'analysis.json',result);ledger('phase2746_autonomous_analysis',result['seconds']);print('SELF_ANALYSIS_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':main()
