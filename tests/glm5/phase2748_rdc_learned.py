"""Actual final learned BF16 deployments, complete teacher/free answers, fixed axes."""
import argparse
import copy
from rdc_question_common import *
from rdc_question_checkpoint import deploy_native
from rdc_question_learning import parameter_fingerprints
from rdc_question_native import QuestionNative
from rdc_question_history import teacher_likelihood,greedy_histories
from phase2748_rdc_native_capture import context_requests,question_requests
from phase2748_rdc_acquire import select_wave,deduplicated_pilot_fields
from rdc_operator_model import memory


def freeze():
    path=OUT/'learned/execution.json'
    rev={'source':snapshot(__file__),'native':snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'history':snapshot(Path(__file__).with_name('rdc_question_history.py')),
        'checkpoint':snapshot(Path(__file__).with_name('rdc_question_checkpoint.py')),
        'retention_sha256':sha(OUT/'retention_contract.json'),
        'all_six_training_result_sha256':sha(OUT/'training/result.json')}
    assert read(OUT/'training/result.json')['all_passed']
    if path.exists():
        value=read(path);assert value['execution']==rev;return value
    value={'timestamp':stamp(),'execution':rev,
        'parameters':'Fresh native Q4BF16model, exact actualCUDA-cast weights decoded from each96stepfullcheckpoint. No training FP32bridge installed. Other parameter full-word fingerprints unchanged before/after.',
        'pilot':'Each learnedrun uses the original8fixedtrainingpilotquestions. Complete teacher/free histories repeat in reverse order, every declared full trajectory field and score bit-equal; actual H12/read fields equal original pretrained fields for all pilot questions.',
        'formal':'Every192validation+384diagnostic+256confirmationquestion has full free answer and separate teacher scoring. Shared context and questionsuffix B1shapes identical to original capture.',
        'retention':'Firstprefix postnorm/H12/source read for all832; full firstprefix selectedMLP/hidden/attention fields and every-generation-step allH/post/H12/read only same48predeclaredquestions. Complete per-step IDs/text/positions/vocabulary scores and separate teacherpostnorm all832 retained.',
        'localization_check':'For every question, H12last and block12native read must exactly equal original model because onlyblock16weightschanged. This is a structural locality check, not semantic-mechanism identification.',
        'confirmation':'material(confirmation=True) requires actual sealed freeze certificate; no confirmation observed here before that gate.'}
    immutable(path,value);return value


def forward_wave(model,tok,engine,batch,rows,contract,reverse=False,progress=None):
    by_id={r['question_id']:r for r in rows}
    full=set(contract['capture']['full_history_context_ids'])
    contexts=context_requests('qwen4',batch,rows,model.config,contract)
    for request in contexts:request.update(collect_hidden=False,full_prefix_H=False)
    engine.forward(contexts)
    requests=question_requests('qwen4',batch,rows,contexts,model.config,reverse)
    for request in requests:
        retain=request['group_id']in full
        request.update(collect_hidden=retain,collect_units=retain,account_attention=retain,save_source_details=retain)
    outputs=engine.forward(requests)
    rr=[by_id[r['question_id']]for r in requests]
    first={r['question_id']:o['fields']for r,o in zip(requests,outputs)}
    teachers,teacher_seconds=teacher_likelihood(model,engine,requests,outputs,rr,progress)
    histories,history_seconds=greedy_histories(model,tok,engine,requests,outputs,rr,contract,progress)
    result={r['question_id']:{'first':first[r['question_id']],'teacher':teacher,'history':history}
        for r,teacher,history in zip(rr,teachers,histories)}
    return result,{'teacher_seconds':teacher_seconds,'history_seconds':history_seconds}


def same_early(arrays,reference):
    assert sha(ROOT/reference['path'])==reference['sha256']
    with np.load(ROOT/reference['path'])as z:
        assert all(np.array_equal(arrays[k],z[k])for k in ['H12_last_BF16','native_source_read_BF16'])


def pilot(model,tok,engine,run,checkpoint,contract,rows,groups):
    path=OUT/'learned'/run/'pilot/result.json'
    if path.exists():
        result=read(path);assert result['all_passed'] and result['checkpoint_sha256']==sha(OUT/'training'/run/'checkpoint96.json');return
    selected=[g for gid in contract['capture']['pilot_context_ids']for g in groups if g['group_id']==gid]
    full=copy.deepcopy(contract)
    full['capture']['full_history_context_ids'] += [g['group_id']for g in selected]
    first,timing=forward_wave(model,tok,engine,selected,rows,full)
    repeated,_=forward_wave(model,tok,engine,selected,rows,full,True)
    references=deduplicated_pilot_fields('qwen4')
    records=[]
    for qid,a in first.items():
        b=repeated[qid]
        assert set(a['first'])==set(b['first']) and all(np.array_equal(a['first'][k],b['first'][k])for k in a['first'])
        same_early(a['first'],references[qid])
        ref=commit_arrays(Path('learned')/run/'pilot/first',qid.replace(':','_'),a['first'])
        record={'question_id':qid,'first_field':ref}
        for kind in ['teacher','history']:
            arrays,rec=a[kind];other,orec=b[kind]
            assert rec==orec and set(arrays)==set(other) and all(np.array_equal(arrays[k],other[k])for k in arrays)
            reference=commit_arrays(Path('learned')/run/'pilot'/kind,qid.replace(':','_'),arrays)
            record[kind]={**rec,'field':reference}
        records.append(record)
    immutable(path,{'timestamp':stamp(),'all_passed':True,'run':run,'questions':len(records),
        'checkpoint_sha256':sha(OUT/'training'/run/'checkpoint96.json'),
        'execution_sha256':sha(OUT/'learned/execution.json'),'records':records,'timing':timing,
        'original_early_fields_exact':True,'reverse_all_fields_scores_exact':True,
        'scope':'Numerical learnedBF16qualification on trainingquestions, not heldout effect.'})
    print('NATURAL_LEARNED_PILOT_PASS',run,len(records),flush=True)


def main(only_run=None,confirmation=False,pilot_only=False):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION)|{'phase2748_rdc_acquire.py','phase2748_rdc_training.py','phase2748_rdc_learning_pilot.py',
        'phase2748_rdc_prospective.py','phase2748_rdc_learned.py','phase2748_rdc_readout.py'})
    freeze()
    names=[r['run']for r in read(OUT/'training/material_manifest.json')['run_inventory']]
    assert only_run is None or only_run in names
    runs=names if only_run is None else[only_run]
    contract,_,pilot_rows,pilot_groups=material('qwen4',False)
    _,_,rows,groups=material('qwen4',confirmation)
    if not confirmation:
        rows=[r for r in rows if r['split']in ['validation','diagnostic']]
        groups=[g for g in groups if g['split']in ['validation','diagnostic']]
    by_id={r['question_id']:r for r in rows}
    scope='confirmation'if confirmation else'nonconfirmation'
    native_groups={g['group_id']:read(OUT/f'native/qwen4/{scope}/groups/{g["group_id"]}.json')for g in groups}
    native_refs={q['question_id']:q['field']for g in native_groups.values()for q in g['questions']}
    start=time.monotonic();model=engine=None
    try:
        model,tok=load('qwen4',OUT/'learned/loaders'/str(time.time_ns()))
        unchanged=parameter_fingerprints(model,exclude_target=True)
        engine=QuestionNative(model,contract['capture']['selected_MLP_blocks']['qwen4'])
        for run in runs:
            folder=Path('learned')/run/scope;final=OUT/folder/'result.json'
            if final.exists():
                assert read(final)['execution_sha256']==sha(OUT/'learned/execution.json');continue
            checkpoint=deploy_native(model,run)
            pilot(model,tok,engine,run,checkpoint,contract,pilot_rows,pilot_groups)
            if pilot_only:continue
            committed=[]
            for group in groups:
                path=OUT/folder/'groups'/(group['group_id']+'.json')
                if path.exists():
                    record=read(path);assert record['checkpoint_sha256']==sha(OUT/'training'/run/'checkpoint96.json')
                    for q in record['questions']:
                        for ref in [q['field'],q['teacher']['field'],q['history']['field']]:assert sha(ROOT/ref['path'])==ref['sha256']
                    committed.append(record)
            done={g['group_id']for g in committed};pending=[g for g in groups if g['group_id']not in done]
            while pending:
                state=memory();assert state['host_available_bytes']>2*1024**3 and state['system_commit_headroom']>2*1024**3,state
                storage_guard();batch,forecast=select_wave(model,pending,by_id)
                last=[time.monotonic()]
                def progress(stage,step,tokens,seconds):
                    if step==1 or step%8==0 or time.monotonic()-last[0]>20:
                        print('NATURAL_LEARNED_HISTORY',run,len(committed),stage,step,tokens,round(seconds,1),flush=True);last[0]=time.monotonic()
                values,timing=forward_wave(model,tok,engine,batch,rows,contract,progress=progress)
                records={}
                for qid,value in values.items():
                    full=by_id[qid]['group_id']in contract['capture']['full_history_context_ids']
                    same_early(value['first'],native_refs[qid])
                    first=value['first']if full else{k:value['first'][k]for k in ['postnorm_BF16','H12_last_BF16','native_source_read_BF16']}
                    ref=commit_arrays(folder/'first',qid.replace(':','_'),first)
                    record={'question_id':qid,'group_id':by_id[qid]['group_id'],'cohort':by_id[qid]['cohort'],
                        'split':by_id[qid]['split'],'field':ref,'original_H12_and_source_read_bit_equal':True}
                    for kind in ['teacher','history']:
                        arrays,rec=value[kind]
                        if kind=='history'and not full:
                            arrays={k:arrays[k]for k in ['generated_ids','statistics','positions']}
                        ref=commit_arrays(folder/kind,qid.replace(':','_'),arrays)
                        record[kind]={**rec,'field':ref,'deployment':'Learned final96stepnativeBF16',
                            'checkpoint_sha256':sha(OUT/'training'/run/'checkpoint96.json'),
                            'internal_trajectory_scope':('Complete given-teacher postnorm and vocabulary scores; not allH teacher trajectories.' if kind=='teacher'
                                else 'AllH/post/H12/read every free-generation step' if full
                                else 'Intermediate free-generation internal trajectory not persisted; firstprefix and complete teacherpostnorm retained separately.')}
                    records[qid]=record
                for group in batch:
                    record={'group_id':group['group_id'],'split':group['split'],'cohort':group['cohort'],
                        'run':run,'checkpoint_sha256':sha(OUT/'training'/run/'checkpoint96.json'),
                        'execution_sha256':sha(OUT/'learned/execution.json'),
                        'questions':[records[qid]for qid in group['four_initial_question_ids']],
                        'wave_forecast':forecast,'wave_timing':timing}
                    immutable(OUT/folder/'groups'/(group['group_id']+'.json'),record);committed.append(record)
                pending=pending[len(batch):]
                del values,records
                gc.collect();torch.cuda.empty_cache()
                save(OUT/folder/'progress.json',{'timestamp':stamp(),'contexts':len(committed),'total_contexts':len(groups),'questions':4*len(committed)})
            assert parameter_fingerprints(model,exclude_target=True)==unchanged
            flat=[q for g in committed for q in g['questions']]
            result={'timestamp':stamp(),'all_passed':True,'run':run,'split_scope':scope,'contexts':len(committed),'questions':len(flat),
                'free_generated_tokens':sum(len(q['history']['generated_ids'])for q in flat),'teacher_tokens':sum(q['teacher']['tokens']for q in flat),
                'all_original_early_fields_exact':True,'all_other_parameter_words_unchanged':True,
                'checkpoint_sha256':sha(OUT/'training'/run/'checkpoint96.json'),'execution_sha256':sha(OUT/'learned/execution.json'),
                'scope':'Actual learnednativeBF16complete behavior under declared retention axes. Scores remain conservative completeanswer, format and stop separate.'}
            immutable(final,result)
            print('NATURAL_LEARNED_COMPLETE',run,scope,len(flat),result['free_generated_tokens'],flush=True)
    except Exception as exc:
        failure(OUT/'learned',start,exc);raise
    finally:
        if engine is not None:engine.close()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run');parser.add_argument('--confirmation',action='store_true');parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args();main(args.run,args.confirmation,args.pilot_only)
