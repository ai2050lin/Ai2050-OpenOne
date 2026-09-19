"""Fixed six-run formation analysis: behavior, teacher scores and full fields."""
import argparse
from collections import defaultdict
from rdc_question_common import *
from phase2748_rdc_fit_analysis import bootstrap
import rdc_question_data as data

COLUMNS=['teacher_mean_NLL','teacher_first_NLL','teacher_later_mean_NLL',
    'native_teacher_mean_NLL','native_teacher_first_NLL','native_teacher_later_mean_NLL',
    'complete_correct_and_stopped','native_complete_correct_and_stopped',
    'strict_JSON','native_strict_JSON','natural_EOS','native_natural_EOS','cap_censored','native_cap_censored',
    'generated_tokens','native_generated_tokens','changed_free_sequence',
    'first_state_change_MSE','within_state_change_MSE','native_within_response_energy',
    'learned_within_response_energy','within_native_learned_innerproduct','teacher_state_change_MSE']
PRIMARY_METRICS=['teacher_mean_NLL','teacher_first_NLL','teacher_later_mean_NLL','complete_correct_and_stopped',
    'strict_JSON','natural_EOS','cap_censored','first_state_change_MSE','within_state_change_MSE','teacher_state_change_MSE']


def state_metrics(native,learned):
    assert native.shape==learned.shape and native.ndim==2 and len(native)==4
    nc=native-native.mean(0);lc=learned-learned.mean(0)
    difference=learned-native;within=lc-nc
    return {'first_state_change_MSE':np.mean(difference**2,axis=1),
        'within_state_change_MSE':np.mean(within**2,axis=1),
        'native_within_response_energy':np.mean(nc**2,axis=1),
        'learned_within_response_energy':np.mean(lc**2,axis=1),
        'within_native_learned_innerproduct':np.mean(nc*lc,axis=1)}, {
        'absolute_state_change_MSE_by_coordinate':np.mean(difference**2,axis=0),
        'within_state_change_MSE_by_coordinate':np.mean(within**2,axis=0)}


def freeze():
    path=OUT/'learning_analysis/execution.json'
    execution={'source':snapshot(__file__),'data':snapshot(Path(__file__).with_name('rdc_question_data.py')),
        'paired_bootstrap':snapshot(Path(__file__).with_name('phase2748_rdc_fit_analysis.py')),
        'training_material_sha256':sha(OUT/'training/material_manifest.json'),
        'effective_contract_sha256':sha(OUT/'effective_experiment_contract.json')}
    if path.exists():
        old=read(path);assert old['execution']==execution;return old
    inventory=read(OUT/'training/material_manifest.json')['run_inventory']
    assert not any(list((OUT/'training'/r['run']/'steps').glob('*.json'))for r in inventory),'Analysis not frozen before formal updates'
    unit=read(OUT/'unit/learning_analysis_current.json')
    assert unit['all_passed']and unit['analysis']['sha256']==execution['source']['sha256']
    value={'timestamp':stamp(),'execution':execution,'unit_sha256':sha(OUT/'unit/learning_analysis_current.json'),'before_any_formal2748_optimizer_step':True,
        'run_names':[r['run']for r in inventory],'columns':COLUMNS,'primary_metrics':PRIMARY_METRICS,
        'outcomes':'Same complete native-B1 given-true-teacher answer includingEOS for every learned condition. Fullanswer mean, first format NLL, later-token mean separately. Free-generation complete match, JSON, EOS and128cap separate.',
        'main_comparisons':'Final96steptrue_complete_answer versus whole-answer-permuted and surface-class controls; each matched seed and arithmetic mean over both seeds at each question before context bootstrap. Also everyrun versus original native.',
        'field_change':'Every firstprefix postnorm coordinate. Four-question within-context centering is retrospective only. Report absolute change, question-specific change, native/learned amplitudes and innerproduct, not call larger/smaller changes semantically better.',
        'teacher_field':'Every given-teacher token/postnorm; same fullteacherIDs must match original. No free-history state alignment assumed after sequence divergence.',
        'bootstrap':'2000whole-context paired resamples within each cohort, seeds2748005/2748006; equalcohortmean; descriptive95percent no multiplicity adjustment. Intervals condition on these two training seeds, not uncertainty over training-seed population.',
        'score_limits':'Conservative literal full-answer match, not official benchmark metric or a complete semantic adjudication. Surface-class loss still sees true content history. Oneblockfine-tuning is newformation evidence, not recovery of pretraining.',
        'confirmation':'Only after gate and all6learnedBF16confirmation plus nativeconfirmation complete. Same code/metrics, no checkpoint or rule selection.'}
    immutable(path,value);return value


def first_divergence(left,right):
    for i,(a,b)in enumerate(zip(left,right)):
        if a!=b:return i
    return min(len(left),len(right))if len(left)!=len(right)else None


def main(confirmation=False):
    spec=freeze();start=time.monotonic()
    scope='confirmation'if confirmation else'nonconfirmation'
    folder=Path('learning_analysis')/scope;final=OUT/folder/'result.json'
    if final.exists():
        assert read(final)['execution_sha256']==sha(OUT/'learning_analysis/execution.json')
        print('NATURAL_LEARNING_ANALYSIS_ALREADY_COMPLETE',scope,flush=True);return
    training=read(OUT/'training/result.json');assert training['all_passed']
    names=spec['run_names'];assert len(names)==6
    for run in names:
        result=read(OUT/'learned'/run/scope/'result.json')
        assert result['all_passed']and result['all_original_early_fields_exact']and result['all_other_parameter_words_unchanged']
        assert result['checkpoint_sha256']==sha(OUT/'training'/run/'checkpoint96.json')
    splits=['confirmation']if confirmation else['validation','diagnostic']
    rows,ng,nq=data.index('qwen4',set(splits))
    assert len(rows)==(256 if confirmation else 576)
    group_rows=defaultdict(list)
    for i,row in enumerate(rows):group_rows[row['group_id']].append((i,row))
    run_matrices={};run_results=[]
    for run in names:
        recordpath=OUT/folder/(run+'.json')
        if recordpath.exists():
            old=read(recordpath);assert old['execution_sha256']==sha(OUT/'learning_analysis/execution.json')
            assert sha(ROOT/old['field']['path'])==old['field']['sha256']
            with np.load(ROOT/old['field']['path'])as z:run_matrices[run]=z['metrics'].copy()
            run_results.append(old);continue
        metrics=np.full((len(rows),len(COLUMNS)),np.nan);coord={};records=[]
        for gid,rr in group_rows.items():
            group=read(OUT/'learned'/run/scope/'groups'/(gid+'.json'))
            assert group['run']==run and len(group['questions'])==4
            lq={q['question_id']:q for q in group['questions']}
            native=np.stack([data.field(nq[r['question_id']]['field'],['postnorm_BF16'])['postnorm_BF16']for _,r in rr])
            learned=np.stack([data.field(lq[r['question_id']]['field'],['postnorm_BF16'])['postnorm_BF16']for _,r in rr])
            state,coordinates=state_metrics(native,learned)
            ident=rr[0][1]['split']+'__'+rr[0][1]['cohort']
            if ident not in coord:coord[ident]={'groups':0,**{k:np.zeros_like(v)for k,v in coordinates.items()}}
            coord[ident]['groups']+=1
            for k,v in coordinates.items():coord[ident][k]+=v
            for j,(position,row)in enumerate(rr):
                qid=row['question_id'];a=nq[qid];b=lq[qid]
                assert b['original_H12_and_source_read_bit_equal']
                at=data.field(a['teacher']['field'],['teacher_ids','NLL','postnorm_BF16'])
                bt=data.field(b['teacher']['field'],['teacher_ids','NLL','postnorm_BF16'])
                assert np.array_equal(at['teacher_ids'],bt['teacher_ids'])and len(at['NLL'])>1
                assert at['postnorm_BF16'].shape==bt['postnorm_BF16'].shape
                ah,bh=a['history'],b['history'];ascore,bscore=ah['score'],bh['score']
                values={k:float(v[j])for k,v in state.items()}
                values.update({'teacher_mean_NLL':float(bt['NLL'].mean()),'teacher_first_NLL':float(bt['NLL'][0]),
                    'teacher_later_mean_NLL':float(bt['NLL'][1:].mean()),'native_teacher_mean_NLL':float(at['NLL'].mean()),
                    'native_teacher_first_NLL':float(at['NLL'][0]),'native_teacher_later_mean_NLL':float(at['NLL'][1:].mean()),
                    'complete_correct_and_stopped':int(bscore['whole_response_exact_and_stopped']),
                    'native_complete_correct_and_stopped':int(ascore['whole_response_exact_and_stopped']),
                    'strict_JSON':int(bscore['strict_nonempty_json_string_array']),'native_strict_JSON':int(ascore['strict_nonempty_json_string_array']),
                    'natural_EOS':int(bh['native_EOS']),'native_natural_EOS':int(ah['native_EOS']),
                    'cap_censored':int(bh['censored']),'native_cap_censored':int(ah['censored']),
                    'generated_tokens':len(bh['generated_ids']),'native_generated_tokens':len(ah['generated_ids']),
                    'changed_free_sequence':int(bh['generated_ids']!=ah['generated_ids']),
                    'teacher_state_change_MSE':float(np.mean((at['postnorm_BF16']-bt['postnorm_BF16'])**2))})
                metrics[position]=[values[k]for k in COLUMNS]
                records.append({'question_id':qid,'group_id':gid,'split':row['split'],'cohort':row['cohort'],
                    'first_free_divergence_index':first_divergence(ah['generated_ids'],bh['generated_ids']),
                    'native_output':ah['generated_text'],'learned_output':bh['generated_text'],
                    'native_first_field':a['field'],'learned_first_field':b['field']})
        assert np.isfinite(metrics).all()
        arrays={'metrics':metrics}
        for ident,aggregate in coord.items():
            for k,v in aggregate.items():
                if k!='groups':arrays[ident+'__'+k]=v/aggregate['groups']
        ref=commit_arrays(folder,run,arrays)
        summary=[]
        for split in splits:
            by_cohort={}
            for cohort in ['drop','quoref']:
                take=np.array([r['split']==split and r['cohort']==cohort for r in rows])
                by_cohort[cohort]={'questions':int(take.sum()),'contexts':int(take.sum()/4),
                    **{name:float(metrics[take,i].mean())for i,name in enumerate(COLUMNS)}}
            by_cohort['equal_cohort']={name:float(np.mean([by_cohort[c][name]for c in ['drop','quoref']]))for name in COLUMNS}
            summary.append({'split':split,'summary':by_cohort})
        outputs=OUT/folder/(run+'_outputs.json.gz');assert not outputs.exists();compressed(outputs,records)
        result={'run':run,'field':ref,'columns':COLUMNS,'summaries':summary,'questions':len(rows),
            'execution_sha256':sha(OUT/'learning_analysis/execution.json'),
            'learned_result_sha256':sha(OUT/'learned'/run/scope/'result.json'),
            'outputs':{'path':outputs.relative_to(ROOT).as_posix(),'sha256':sha(outputs)}}
        immutable(recordpath,result);run_results.append(result);run_matrices[run]=metrics
        print('NATURAL_LEARNING_ANALYSIS_RUN',run,scope,round(time.monotonic()-start,1),flush=True)
    # Baseline identity must agree in every run, not count native six times.
    baseline_columns=[i for i,c in enumerate(COLUMNS)if c.startswith('native_')]
    baseline=run_matrices[names[0]][:,baseline_columns]
    assert all(np.array_equal(v[:,baseline_columns],baseline)for v in run_matrices.values())
    comparisons=[]
    for split in splits:
        take=np.array([r['split']==split for r in rows]);rr=[r for r in rows if r['split']==split]
        for run in names:
            array=run_matrices[run][take]
            for metric in PRIMARY_METRICS:
                if 'native_'+metric not in COLUMNS:continue
                comparisons.append({'split':split,'left':run,'right':'native','metric':metric,
                    'paired':bootstrap(rr,array[:,COLUMNS.index(metric)],array[:,COLUMNS.index('native_'+metric)],2748005)})
        for control in ['within_context_permuted_complete_answer','surface_class_mass_on_true_teacher_history']:
            for seeds in [[2748],[2749],[2748,2749]]:
                left=np.mean([run_matrices['true_complete_answer_'+str(s)][take]for s in seeds],axis=0)
                right=np.mean([run_matrices[control+'_'+str(s)][take]for s in seeds],axis=0)
                for metric in PRIMARY_METRICS:
                    comparisons.append({'split':split,'left':'true_complete_answer','right':control,'fixed_seeds':seeds,'metric':metric,
                        'paired':bootstrap(rr,left[:,COLUMNS.index(metric)],right[:,COLUMNS.index(metric)],2748005)})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'split_scope':scope,
        'execution_sha256':sha(OUT/'learning_analysis/execution.json'),'runs':run_results,'paired_comparisons':comparisons,
        'identities':[{k:r[k]for k in ['question_id','group_id','cohort','split','within_context_index']}for r in rows],
        'fixed_native_baseline_identical_across_all6runs':True,'seconds':time.monotonic()-start,
        'interpretation':'Lower teacherNLL is better, higher completecorrect fraction is better. State-change magnitude alone has no improvement direction. No matching of free-history states after histories diverge. Context CIs condition on two fixed training seeds.'}
    immutable(final,result)
    print('NATURAL_LEARNING_ANALYSIS_COMPLETE',scope,len(comparisons),round(result['seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--confirmation',action='store_true');parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args();freeze()if args.freeze_only else main(args.confirmation)
