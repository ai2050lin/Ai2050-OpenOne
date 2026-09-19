"""Fixed own-history comparison; never align states after causal prefixes split."""
import argparse
from collections import Counter
from rdc_question_common import *
import rdc_question_data as data
from phase2748_rdc_fit_analysis import bootstrap

OUTCOMES=['complete_correct_and_stopped','strict_JSON','natural_EOS','cap_censored','generated_tokens']
AGREEMENT=['first_token_matches_native','complete_sequence_matches_native','common_output_prefix_tokens',
    'common_causal_prediction_steps','identical_prefix_fraction_of_longer_output']
COLUMNS=OUTCOMES+['native_'+c for c in OUTCOMES]+AGREEMENT


def sequence_alignment(left,right):
    assert len(left)>0 and len(right)>0
    common=0
    while common<min(len(left),len(right))and left[common]==right[common]:common+=1
    identical=left==right
    divergence=None if identical else common
    # At the first differing output, both states still have the same past.
    # Do not invent the next prediction after either recorded sequence ended.
    valid=min(common+1,len(left),len(right))
    return {'first_token_matches_native':int(left[0]==right[0]),
        'complete_sequence_matches_native':int(identical),'common_output_prefix_tokens':common,
        'common_causal_prediction_steps':valid,'first_output_divergence_index':divergence,
        'identical_prefix_fraction_of_longer_output':common/max(len(left),len(right))}


def outcomes(history):
    return {'complete_correct_and_stopped':int(history['score']['whole_response_exact_and_stopped']),
        'strict_JSON':int(history['score']['strict_nonempty_json_string_array']),
        'natural_EOS':int(history['native_EOS']),'cap_censored':int(history['censored']),
        'generated_tokens':len(history['generated_ids'])}


def freeze():
    path=OUT/'prospective_analysis/execution.json'
    execution={'source':snapshot(__file__),'data':snapshot(Path(__file__).with_name('rdc_question_data.py')),
        'bootstrap':snapshot(Path(__file__).with_name('phase2748_rdc_fit_analysis.py')),
        'material_sha256':sha(OUT/'material/manifest.json'),'retention_sha256':sha(OUT/'retention_contract.json')}
    if path.exists():
        previous=read(path);assert previous['execution']==execution;return previous
    assert not any((OUT/'prospective').glob('**/groups/*.json')),'Freeze before formal own histories'
    unit=read(OUT/'unit/prospective_analysis_current.json')
    assert unit['all_passed']and unit['analysis']['sha256']==execution['source']['sha256']
    value={'timestamp':stamp(),'execution':execution,'unit_sha256':sha(OUT/'unit/prospective_analysis_current.json'),
        'columns':COLUMNS,'before_any_formal_own_history':True,
        'population':'All384diagnostic or all256sealedconfirmation questions, separately by model. Primary and target-pair-shuffled control frozen from original first-prefix validation.',
        'comparisons':'Every rule versus original native for5behavior outcomes; primary versus control for5outcomes and5sequence agreements. Paired2000context bootstrap with original seeds2748005/2748006; descriptive95percent intervals, no multiplicity correction.',
        'state_alignment':'Only predeclared fullyretained questions. Include prediction at first emitted-token divergence because past still equal; exclude every subsequent step. H12/read must equal original native exactly on all valid coordinates/steps. Report prediction MSE only on this prefix-conditional subset, not entire free histories.',
        'weighting':'Behavior perquestion then whole4questioncontext then cohort mean, bothcohorts equal. Prefix state errors first meansteps within each retained question then average retained questions; coordinate aggregation separately percohort. Prefix-survival selection prevents interpreting these as unconditional later-step errors.',
        'limits':'Complete literal scoring is not full semantic adjudication. No same-prefix state error is defined after histories diverge. Current early-only input still runs13original native blocks and uses original full vocabulary head; not standalone reconstruction from parameters alone.',
        'confirmation':'Same source after gate; no retuning or choosing a different rule based on diagnostic.'}
    immutable(path,value);return value


def main(key,confirmation=False):
    start=time.monotonic();spec=freeze();scope='confirmation'if confirmation else'diagnostic'
    folder=Path('prospective_analysis')/key/scope;final=OUT/folder/'result.json'
    if final.exists():
        previous=read(final);assert previous['execution_sha256']==sha(OUT/'prospective_analysis/execution.json')
        print('NATURAL_PROSPECTIVE_ANALYSIS_ALREADY_COMPLETE',key,scope,flush=True);return
    selection=read(OUT/'fit'/key/'validation_selection.json')
    variants=[selection['primary_rule'],selection['control_rule']]
    for variant in variants:
        result=read(OUT/'prospective'/key/variant/scope/'result.json')
        assert result['all_passed']and result['original_late_layers_or_finalnorm_calls']==0
        assert result['questions']==(256 if confirmation else 384)
        assert result['execution_sha256']==sha(OUT/'prospective'/key/'execution.json')
    rows,groups,native=data.index(key,{scope});assert len(rows)==(256 if confirmation else 384)
    full=set(read(OUT/'effective_experiment_contract.json')['capture']['full_history_context_ids'])
    results=[];matrices={}
    for variant in variants:
        resultpath=OUT/folder/(variant+'.json')
        if resultpath.exists():
            old=read(resultpath);assert old['execution_sha256']==sha(OUT/'prospective_analysis/execution.json')
            assert old['prospective_result_sha256']==sha(OUT/'prospective'/key/variant/scope/'result.json')
            for ref in [old['field'],old['outputs']]:assert sha(ROOT/ref['path'])==ref['sha256']
            with np.load(ROOT/old['field']['path'])as z:matrices[variant]=z['metrics'].copy()
            assert matrices[variant].shape==(len(rows),len(COLUMNS))
            results.append(old);continue
        records=[];matrix=np.zeros((len(rows),len(COLUMNS)));coordinate={}
        packets={}
        for gid in groups:
            path=OUT/'prospective'/key/variant/scope/'groups'/(gid+'.json')
            group=read(path);assert group['group_id']==gid and group['split']==scope and len(group['questions'])==4
            assert group['execution_sha256']==sha(OUT/'prospective'/key/'execution.json')
            for q in group['questions']:
                assert q['question_id']not in packets;packets[q['question_id']]=q
        assert set(packets)=={r['question_id']for r in rows}
        for index,row in enumerate(rows):
            qid=row['question_id'];original=native[qid]['history'];own=packets[qid]
            assert own['full_coordinate_trajectories_retained']==(row['group_id']in full)
            alignment=sequence_alignment(original['generated_ids'],own['generated_ids'])
            ownvalues=outcomes(own);nativevalues=outcomes(original)
            values={**ownvalues,**{'native_'+k:v for k,v in nativevalues.items()},**alignment}
            matrix[index]=[values[k]for k in COLUMNS]
            assert sha(ROOT/own['field']['path'])==own['field']['sha256']
            with np.load(ROOT/own['field']['path'])as z:
                assert z['generated_ids'].tolist()==own['generated_ids']
                assert np.array_equal(z['positions'],np.arange(len(own['generated_ids']))+len(row['tokens']['input_ids'])-1)
                assert np.isfinite(z['statistics']).all()and np.isfinite(z['casting_statistics']).all()
            record={k:row[k]for k in ['question_id','group_id','cohort','split','within_context_index']}
            record.update({'alignment':alignment,'native_output':original['generated_text'],'own_output':own['generated_text'],
                'native_field':original['field'],'own_field':own['field'],'same_prefix_state_checked':False})
            if own['full_coordinate_trajectories_retained']:
                names=['H12_last_BF16','native_source_read_BF16']
                aa=data.field(original['field'],names+['postnorm_BF16'])
                bb=data.field(own['field'],names+['predicted_postnorm_FP64','predicted_postnorm_BF16'])
                count=alignment['common_causal_prediction_steps']
                for name in names:assert np.array_equal(aa[name][:count],bb[name][:count]),(qid,name,count)
                errors={p:np.mean((bb['predicted_postnorm_'+p][:count]-aa['postnorm_BF16'][:count])**2,axis=0)for p in ['FP64','BF16']}
                if row['cohort']not in coordinate:coordinate[row['cohort']]={'questions':0,'steps':0,**{p:np.zeros_like(v)for p,v in errors.items()}}
                target=coordinate[row['cohort']];target['questions']+=1;target['steps']+=count
                for p,v in errors.items():target[p]+=v
                record.update({'same_prefix_state_checked':True,'same_prefix_H12_and_read_all_coordinates_exact':True,
                    'same_prefix_unrounded_MSE':float(errors['FP64'].mean()),'same_prefix_BF16_MSE':float(errors['BF16'].mean()),
                    'unconditional_own_history_state_error':None})
            records.append(record)
        assert sum(v['questions']for v in coordinate.values())==sum(r['group_id']in full for r in rows)>0
        assert np.isfinite(matrix).all();arrays={'metrics':matrix}
        for cohort,values in coordinate.items():
            for p in ['FP64','BF16']:arrays[cohort+'__same_prefix_'+p+'_MSE_by_coordinate']=values[p]/values['questions']
        ref=commit_arrays(folder,variant,arrays)
        path=OUT/folder/(variant+'_outputs.json.gz');assert not path.exists();compressed(path,records)
        summaries={}
        for cohort in ['drop','quoref']:
            take=np.array([r['cohort']==cohort for r in rows]);subset=[r for r in records if r['cohort']==cohort]
            summaries[cohort]={'questions':int(take.sum()),'contexts':int(take.sum()/4),
                'means':{c:float(matrix[take,i].mean())for i,c in enumerate(COLUMNS)},
                'first_divergence_histogram':dict(Counter('identical'if r['alignment']['first_output_divergence_index']is None else str(r['alignment']['first_output_divergence_index'])for r in subset))}
        summaries['equal_cohort']={'means':{c:float(np.mean([summaries[h]['means'][c]for h in ['drop','quoref']]))for c in COLUMNS}}
        result={'variant':variant,'field':ref,'columns':COLUMNS,'summaries':summaries,
            'execution_sha256':sha(OUT/'prospective_analysis/execution.json'),
            'prospective_result_sha256':sha(OUT/'prospective'/key/variant/scope/'result.json'),
            'outputs':{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)},
            'same_prefix_state_subset':[{'cohort':c,'questions':v['questions'],'steps':v['steps'],
                **{p+'_MSE':float((v[p]/v['questions']).mean())for p in ['FP64','BF16']}}for c,v in coordinate.items()],
            'all_retained_same_prefix_early_fields_exact':True}
        immutable(resultpath,result);results.append(result);matrices[variant]=matrix
    baseline_columns=[COLUMNS.index('native_'+c)for c in OUTCOMES]
    assert np.array_equal(matrices[variants[0]][:,baseline_columns],matrices[variants[1]][:,baseline_columns])
    comparisons=[]
    for variant in variants:
        for metric in OUTCOMES:
            comparisons.append({'left':variant,'right':'original_native','metric':metric,
                'paired':bootstrap(rows,matrices[variant][:,COLUMNS.index(metric)],matrices[variant][:,COLUMNS.index('native_'+metric)],2748005)})
    for metric in OUTCOMES+AGREEMENT:
        comparisons.append({'left':variants[0],'right':variants[1],'metric':metric,
            'paired':bootstrap(rows,matrices[variants[0]][:,COLUMNS.index(metric)],matrices[variants[1]][:,COLUMNS.index(metric)],2748005)})
    result={'timestamp':stamp(),'all_passed':True,'model':key,'scope':scope,'questions':len(rows),'rules':results,
        'execution_sha256':sha(OUT/'prospective_analysis/execution.json'),'fit_selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
        'paired_comparisons':comparisons,'seconds':time.monotonic()-start,'limits':spec['limits']}
    immutable(final,result);print('NATURAL_PROSPECTIVE_ANALYSIS_COMPLETE',key,scope,round(result['seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen4','qwen14','glm4'])
    parser.add_argument('--confirmation',action='store_true');parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:freeze()
    else:
        assert args.model;main(args.model,args.confirmation)
