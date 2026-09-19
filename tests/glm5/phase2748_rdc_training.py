"""Six actual complete-answer training runs; fixed96step full-parameter endpoint."""
import argparse
from rdc_question_common import *
from rdc_question_learning import *
from rdc_question_checkpoint import original_manifest, save_final
from rdc_question_native import QuestionNative
from rdc_question_history import teacher_likelihood
from phase2748_rdc_native_capture import context_requests, question_requests
from phase2748_rdc_acquire import select_wave
from rdc_operator_model import memory


def protocol():
    path = OUT/'training/formal_protocol.json'
    rev = {'source':snapshot(__file__),'learning':snapshot(Path(__file__).with_name('rdc_question_learning.py')),
        'checkpoint':snapshot(Path(__file__).with_name('rdc_question_checkpoint.py')),
        'execution_contract_sha256':sha(OUT/'training/execution_contract.json'),
        'training_material_sha256':sha(OUT/'training/material_manifest.json'),
        'pilot_sha256':sha(OUT/'training/pilot/result.json')}
    pilot = read(OUT/'training/pilot/result.json')
    assert pilot['all_passed'] and pilot['execution_contract_sha256'] == rev['execution_contract_sha256']
    if path.exists():
        old = read(path)
        assert old['execution'] == rev,'Formal source changed; versioned repair required'
        return old
    value = {'timestamp':stamp(),'execution':rev,'runs':read(OUT/'training/material_manifest.json')['run_inventory'],
        'checkpoints':[1,8,32,96],'final_step':96,'step_size':.02,
        'resume':'Completed full96step runs skip after verifying their full parameter archives. An interrupted run starts from unchanged original bridge and repeats seeded draws; existing step scores/fields must agree. No intermediate parameter checkpoint is falsely implied.',
        'validation_shape':'Use the qualified common-prefix segmented B1 engine and sequential given-teacher tokens under the active FP32MLPbridge, not joint teacher training shape.',
        'learned_behavior':'Final nativeBF16deployment complete histories are a separate stage. No confirmation target read in this training process.',
        'resource_guards':'AllCUDAmodels sequential, positive host/commit headroom and4GiB on each drive; measured pilot peak. No model parameter file is overwritten.'}
    immutable(path,value)
    return value


def validate(model, engine, rows, groups, contract, run, step):
    import torch
    target = OUT/'training'/run/'validation'/('step'+str(step)+'.json')
    by_id = {r['question_id']:r for r in rows}
    pending = list(groups)
    records = []
    tick = time.monotonic()
    while pending:
        batch, forecast = select_wave(model,pending,by_id)
        contexts = context_requests('qwen4',batch,rows,model.config,contract)
        for request in contexts:
            request.update(collect_hidden=False,full_prefix_H=False)
        engine.forward(contexts)
        requests = question_requests('qwen4',batch,rows,contexts,model.config)
        for request in requests:
            request.update(collect_hidden=False,collect_units=False,account_attention=False,save_source_details=False)
        outputs = engine.forward(requests)
        rr = [by_id[r['question_id']] for r in requests]
        packets,_ = teacher_likelihood(model,engine,requests,outputs,rr)
        for arrays,record in packets:
            qid = record['question_id']
            reference = commit_arrays(Path('training')/run/'validation'/('step'+str(step)),qid.replace(':','_'),arrays)
            records.append({**record,'field':reference})
        pending = pending[len(batch):]
        del contexts,requests,outputs,packets
        gc.collect(); torch.cuda.empty_cache()
        print('NATURAL_TRAIN_VALIDATION',run,step,len(records),192,flush=True)
    assert len(records)==192 and len({r['question_id'] for r in records})==192
    records.sort(key=lambda r:r['question_id'])
    result = {'run':run,'step':step,'questions':192,'teacher_tokens':sum(r['tokens'] for r in records),
        'records':records,'mean_answer_normalized_NLL':float(np.mean([r['mean_token_NLL'] for r in records])),
        'execution':'Original otherBF16parameters, active fullFP32block16bridge, segmentedB1sequential teacher likelihood. Not free generation or checkpoint selection.'}
    # Timestamps and measured runtimes live in a separate attempt receipt so an
    # exact repeated seeded run can verify existing scientific arrays/records.
    immutable(target,result)
    immutable(OUT/'training'/run/'validation'/('step'+str(step)+'_attempt_'+str(time.time_ns())+'.json'),
        {'timestamp':stamp(),'seconds':time.monotonic()-tick,'result_sha256':sha(target)})
    return result


def verify_finished(run):
    receipt = read(OUT/'training'/run/'checkpoint96.json')
    assert receipt['all_passed'] and receipt['total_parameters']==74711040
    for parameter in receipt['parameters']:
        assert sha(ROOT/parameter['field']['path']) == parameter['field']['sha256']
    result = read(OUT/'training'/run/'result.json')
    assert result['checkpoint96_sha256'] == sha(OUT/'training'/run/'checkpoint96.json')
    return result


def main(only_run=None):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION)|{'phase2748_rdc_acquire.py','phase2748_rdc_training.py',
        'phase2748_rdc_learning_pilot.py','phase2748_rdc_prospective.py','phase2748_rdc_learned.py','phase2748_rdc_readout.py'})
    proto = protocol()
    contract,manifest,rows,groups = material('qwen4')
    training = read(OUT/'training/material_manifest.json')
    all_draws = gzread(ROOT/training['draws']['path'])
    by_id = {r['question_id']:r for r in rows}
    validation_rows = [r for r in rows if r['split']=='validation']
    validation_groups = [g for g in groups if g['split']=='validation']
    names = [r['run'] for r in proto['runs']]
    assert only_run is None or only_run in names
    selected = names if only_run is None else [only_run]
    pending = [run for run in selected if not (OUT/'training'/run/'result.json').exists()]
    for run in selected:
        if run not in pending:
            verify_finished(run)
    if not pending:
        print('NATURAL_TRAINING_SELECTED_RUNS_ALREADY_COMPLETE',selected,flush=True)
        return
    start = time.monotonic()
    model=engine=None
    handles=[]
    try:
        original_manifest()
        model,tok=load('qwen4',OUT/'training'/'formal_loader'/str(time.time_ns()))
        assert all(p.device.type=='cuda' for p in model.parameters())
        unchanged = parameter_fingerprints(model,exclude_target=True)
        unchangedpath=OUT/'training/unchanged_native_parameters.json'
        immutable(unchangedpath,unchanged)
        target,original,handles=prepare_gradients(model)
        params=list(target.parameters())
        assert sum(p.numel() for p in params)==74711040
        engine=QuestionNative(model,[])
        with np.load(ROOT/training['partition_and_exposure']['path']) as z:
            classes=torch.tensor(z['classes'].astype(np.int64),device='cuda')
        for run in pending:
            guard();storage_guard()
            restore(target,original)
            draws=[d for d in all_draws if d['run']==run]
            assert len(draws)==768 and [d['draw'] for d in draws]==list(range(768))
            assert all(by_id[d['question_id']]['split']=='train' for d in draws)
            records=[]
            tick=time.monotonic()
            for step in range(1,97):
                state=memory()
                assert state['host_available_bytes']>2*1024**3 and state['system_commit_headroom']>2*1024**3, state
                batch=draws[8*(step-1):8*step]
                grads,losses=mean_gradient(model,by_id,batch,classes,params)
                norm=full_norm(grads)
                assert np.isfinite(norm) and norm>0
                with torch.no_grad():
                    for p,g in zip(params,grads):
                        p.add_(g,alpha=-.02/(norm+1e-12))
                assert all(bool(torch.isfinite(p).all()) for p in params)
                del grads
                record={'step':step,'question_ids':[d['question_id'] for d in batch],
                    'teacher_from_question_ids':[d['teacher_from_question_id'] for d in batch],
                    'teacher_token_count':sum(len(d['teacher_ids_including_EOS']) for d in batch),
                    'per_complete_answer_loss':losses,'mean_batch_loss':float(np.mean(losses)),
                    'full_parameter_gradient_norm':norm,'step_nominal_FP32_L2':.02}
                immutable(OUT/'training'/run/'steps'/(str(step)+'.json'),record)
                records.append(record)
                print('NATURAL_TRAINING_STEP',run,step,96,round(record['mean_batch_loss'],6),round(time.monotonic()-tick,1),flush=True)
                if step in proto['checkpoints']:
                    delta_norm=full_norm([p.detach()-original[n].to(p.device) for n,p in target.named_parameters()])
                    summary={'run':run,'step':step,'full_parameter_change_L2':delta_norm,
                        'parameter_L2':full_norm(params),'all74711040parameters_used':True}
                    immutable(OUT/'training'/run/'parameter_summaries'/(str(step)+'.json'),summary)
                    validate(model,engine,validation_rows,validation_groups,contract,run,step)
            checkpoint=save_final(target,run,sha(OUT/'training/formal_protocol.json'))
            assert parameter_fingerprints(model,exclude_target=True)==unchanged
            result={'timestamp':stamp(),'run':run,'all_passed':True,'steps':96,'questions':768,
                'teacher_tokens':sum(r['teacher_token_count'] for r in records),
                'checkpoint96_sha256':sha(OUT/'training'/run/'checkpoint96.json'),
                'checkpoint96_bytes':checkpoint['bytes'],'all_other_parameters_word_identical':True,
                'unchanged_parameter_receipt_sha256':sha(unchangedpath),
                'formal_protocol_sha256':sha(OUT/'training/formal_protocol.json'),
                'seconds':time.monotonic()-tick,'peak_CUDA_bytes':torch.cuda.max_memory_allocated(),
                'status':'Final96stepfullFP32/BF16parameters and4complete validation teacher checkpoints done. NativeBF16behavior/confirmation are separate.'}
            immutable(OUT/'training'/run/'result.json',result)
            print('NATURAL_TRAINING_RUN_COMPLETE',run,round(result['seconds'],1),flush=True)
        if all((OUT/'training'/run/'result.json').exists() for run in names):
            finished=[verify_finished(run) for run in names]
            immutable(OUT/'training/result.json',{'timestamp':stamp(),'all_passed':True,
                'runs':finished,'formal_protocol_sha256':sha(OUT/'training/formal_protocol.json'),
                'actual_optimizer_steps':6*96,'actual_training_question_draws':6*768,
                'all_final_checkpoints_fixed_step96':True,'confirmation_observed':False})
    except Exception as exc:
        failure(OUT/'training',start,exc)
        raise
    finally:
        if engine is not None:engine.close()
        for handle in handles:handle.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run')
    main(parser.parse_args().run)
