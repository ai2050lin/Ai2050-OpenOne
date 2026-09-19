"""Frozen early-only predictors advance their own real generated histories."""
import argparse
import copy
from rdc_question_common import *
from rdc_question_predictor import FrozenPredictor,early_contexts,early_questions,own_histories,early_cache_id
from rdc_question_native import QuestionNative
from phase2748_rdc_acquire import deduplicated_pilot_fields
from rdc_operator_model import memory


def freeze(key):
    folder=OUT/'prospective'/key
    path=folder/'execution.json'
    unit=read(OUT/'unit'/('predictor_'+key+'_current.json'))
    cache_unit=read(OUT/'unit/early_cache_current.json')
    helper=Path(__file__).with_name('rdc_question_predictor.py')
    assert unit['all_passed'] and unit['predictor']['sha256']==sha(helper)
    assert cache_unit['all_passed'] and cache_unit['predictor']['sha256']==sha(helper)
    rev={'source':snapshot(__file__),'predictor':snapshot(helper),
        'native':snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'coefficient_unit_sha256':sha(OUT/'unit'/('predictor_'+key+'_current.json')),
        'early_cache_unit_sha256':sha(OUT/'unit/early_cache_current.json'),
        'selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
        'retention_sha256':sha(OUT/'retention_contract.json'),
        'effective_contract_sha256':sha(OUT/'effective_experiment_contract.json')}
    if path.exists():
        value=read(path);assert value['execution']==rev;return value
    value={'timestamp':stamp(),'model':key,'execution':rev,
        'early_path':'Only original native blocks0through12execute. Hook each later block to fail if called. Native H12last and block12attentionwrite are supplied to the frozen full-coordinate predictor; native finalnorm is never called by early engine.',
        'pilot':'The8previously fixed training native-pilot questions. Match earlycontextmeans,H12last,source read to retained original full model; then each frozen rule complete128cap nativeEOSownhistories and reverse-request replay every stored field. Pilot temporarily retains full early/predicted fields for all8trainingquestions; formal retention remains the predeclared32diagnostic/confirmationquestions.',
        'formal':'Two frozen rules,384diagnostic and256confirmationquestions per rule/model, own past only. Per-context complete4question commits. No answer-target input, no later native state.',
        'CPU_precision':'Saved float64coefficients; one prediction row per question/step. CPUcoefficient unit measures its roundoff against original192-row validation matmul.',
        'resource':'Up to16independentB1contexts/wave with13-layer KVcache budget, actualfreeCUDA and1GiBworkspace plus1GiBcurrentlayer onlargemodels. All128stepscharged even if some histories stop early. Source weight residency unchanged from originalpilot.'}
    immutable(path,value);return value


def wave(model,pending,by_id):
    import torch
    torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info()
    chosen=[];forecast=None
    per_token=2*13*model.config.num_key_value_heads*model.config.head_dim*2
    for group in pending[:16]:
        if group['split']!=pending[0]['split']:break
        candidate=chosen+[group]
        source=sum(len(by_id[g['four_initial_question_ids'][0]]['tokens']['context_prefix_ids'])for g in candidate)
        prompt=sum(len(by_id[q]['tokens']['input_ids'])for g in candidate for q in g['four_initial_question_ids'])
        cache=(source+prompt+128*4*len(candidate))*per_token
        reserve=source*model.config.hidden_size*2*4+1024**3+(0 if model.config.hidden_size==2560 else 1024**3)
        if cache+reserve>free-256*1024**2:break
        chosen=candidate;forecast={'contexts':len(candidate),'cache_bytes_all128steps':cache,'workspace_and_activation_bytes':reserve,
            'actual_free_CUDA':free,'executed_cache_layers':13,'B1arithmetic_unchanged':True}
    assert chosen,('No safe early-onlyonecontextwave',free)
    return chosen,forecast


def pilot(key,model,tok,engine,contract,rows,groups,rules):
    path=OUT/'prospective'/key/'pilot/result.json'
    if path.exists():
        value=read(path);assert value['all_passed'] and value['execution_sha256']==sha(OUT/'prospective'/key/'execution.json');return value
    start=time.monotonic()
    selected=[g for gid in contract['capture']['pilot_context_ids']for g in groups if g['group_id']==gid]
    by_id={r['question_id']:r for r in rows}
    rr=[by_id[q]for g in selected for q in g['four_initial_question_ids']]
    references=deduplicated_pilot_fields(key)
    contexts,means=early_contexts(key,selected,rows,model,engine,contract)
    for gid,value in means.items():
        ref=references[gid];assert sha(ROOT/ref['path'])==ref['sha256']
        with np.load(ROOT/ref['path'])as z:assert np.array_equal(value,z['context_H12_mean'])
    initial=early_cache_id(contexts[0]['cache']),early_cache_id(contexts[1]['cache'])
    requests,outputs=early_questions(key,selected,rows,contexts,model,engine)
    for request,output in zip(requests,outputs):
        ref=references[request['question_id']];assert sha(ROOT/ref['path'])==ref['sha256']
        with np.load(ROOT/ref['path'])as z:
            assert all(np.array_equal(output['fields'][k],z[k])for k in ['H12_last_BF16','native_source_read_BF16'])
    del requests,outputs
    extended=copy.deepcopy(contract)
    extended['capture']['full_history_context_ids']+= [g['group_id']for g in selected]
    records=[]
    for rule in rules:
        requests,outputs=early_questions(key,selected,rows,contexts,model,engine)
        packets,elapsed=own_histories(model,tok,engine,rule,requests,outputs,rr,means,extended)
        requests2,outputs2=early_questions(key,selected,rows,contexts,model,engine,reverse=True)
        reverse_rows=[by_id[r['question_id']]for r in requests2]
        repeated,_=own_histories(model,tok,engine,rule,requests2,outputs2,reverse_rows,means,extended)
        repeat={r['question_id']:(a,r)for a,r in repeated}
        for arrays,record in packets:
            other,orecord=repeat[record['question_id']]
            assert record==orecord and set(arrays)==set(other)
            assert all(np.array_equal(arrays[k],other[k])for k in arrays)
            ref=commit_arrays(Path('prospective')/key/'pilot'/rule.variant,record['question_id'].replace(':','_'),arrays)
            records.append({**record,'field':ref})
        assert initial==(early_cache_id(contexts[0]['cache']),early_cache_id(contexts[1]['cache']))
        del requests,outputs,requests2,outputs2,packets,repeated,repeat
        print('NATURAL_PROSPECTIVE_PILOT_RULE',key,rule.variant,round(elapsed,1),flush=True)
    result={'timestamp':stamp(),'all_passed':True,'model':key,'execution_sha256':sha(OUT/'prospective'/key/'execution.json'),
        'early_native_fields_exact_questions':8,'rules':len(rules),'records':records,
        'source13layercaches_unchanged':True,'reverse_all_history_fields_exact':True,'seconds':time.monotonic()-start,
        'scope':'Engineering/scheduling qualification on trainingquestions, not heldout language evidence.'}
    immutable(path,result);return result


def main(key,confirmation=False,pilot_only=False):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION)|{'phase2748_rdc_acquire.py','phase2748_rdc_training.py','phase2748_rdc_learning_pilot.py',
        'phase2748_rdc_prospective.py','phase2748_rdc_learned.py','phase2748_rdc_readout.py'})
    execution=freeze(key)
    contract,_,rows,groups=material(key,False)
    selection=read(OUT/'fit'/key/'validation_selection.json')
    rules=[FrozenPredictor(key,v)for v in [selection['primary_rule'],selection['control_rule']]]
    start=time.monotonic();model=engine=None;handles=[]
    try:
        model,tok=load(key,OUT/'prospective'/key/'loaders'/str(time.time_ns()),gpu_limit=6 if key=='qwen14' else 11 if key=='glm4' else None)
        engine=QuestionNative(model,[])
        def forbidden(m,a):
            raise AssertionError('Original late decoder or finalnormcalled in early-onlypredictor')
        handles=[layer.register_forward_pre_hook(forbidden)for layer in model.model.layers[13:]]
        handles.append(model.model.norm.register_forward_pre_hook(forbidden))
        pilot_result=pilot(key,model,tok,engine,contract,rows,groups,rules)
        if pilot_only:return
        if confirmation:
            contract,_,rows,groups=material(key,True)
        else:
            rows=[r for r in rows if r['split']=='diagnostic'];groups=[g for g in groups if g['split']=='diagnostic']
        scope='confirmation'if confirmation else'diagnostic'
        by_id={r['question_id']:r for r in rows}
        for rule in rules:
            folder=Path('prospective')/key/rule.variant/scope
            final=OUT/folder/'result.json'
            if final.exists():
                assert read(final)['execution_sha256']==sha(OUT/'prospective'/key/'execution.json');continue
            committed=[]
            for group in groups:
                path=OUT/folder/'groups'/(group['group_id']+'.json')
                if path.exists():
                    record=read(path);assert record['predictor']==rule.identity
                    for q in record['questions']:assert sha(ROOT/q['field']['path'])==q['field']['sha256']
                    committed.append(record)
            done={r['group_id']for r in committed};pending=[g for g in groups if g['group_id']not in done]
            while pending:
                state=memory();assert state['host_available_bytes']>2*1024**3 and state['system_commit_headroom']>2*1024**3,state
                storage_guard();batch,forecast=wave(model,pending,by_id)
                contexts,means=early_contexts(key,batch,rows,model,engine,contract)
                requests,outputs=early_questions(key,batch,rows,contexts,model,engine)
                rr=[by_id[r['question_id']]for r in requests]
                last=[time.monotonic()]
                def progress(step,tokens,seconds):
                    if step==1 or step%8==0 or time.monotonic()-last[0]>20:
                        print('NATURAL_PROSPECTIVE_HISTORY',key,rule.variant,len(committed),step,tokens,round(seconds,1),flush=True);last[0]=time.monotonic()
                packets,elapsed=own_histories(model,tok,engine,rule,requests,outputs,rr,means,contract,progress)
                qrecords={}
                for arrays,record in packets:
                    qid=record['question_id'];ref=commit_arrays(folder/'histories',qid.replace(':','_'),arrays)
                    qrecords[qid]={**record,'field':ref}
                for group in batch:
                    record={'group_id':group['group_id'],'cohort':group['cohort'],'split':scope,'predictor':rule.identity,
                        'execution_sha256':sha(OUT/'prospective'/key/'execution.json'),
                        'questions':[qrecords[q]for q in group['four_initial_question_ids']],
                        'wave_forecast':forecast,'wave_history_seconds':elapsed}
                    immutable(OUT/folder/'groups'/(group['group_id']+'.json'),record);committed.append(record)
                pending=pending[len(batch):]
                del contexts,means,requests,outputs,packets,qrecords
                gc.collect();torch.cuda.empty_cache()
                save(OUT/folder/'progress.json',{'timestamp':stamp(),'contexts':len(committed),'total_contexts':len(groups),'questions':4*len(committed)})
            flat=[q for g in committed for q in g['questions']]
            result={'timestamp':stamp(),'all_passed':True,'model':key,'variant':rule.variant,'split':scope,
                'contexts':len(committed),'questions':len(flat),'generated_tokens':sum(len(q['generated_ids'])for q in flat),
                'full_coordinate_trajectory_questions':sum(q['full_coordinate_trajectories_retained']for q in flat),
                'predictor':rule.identity,'execution_sha256':sha(OUT/'prospective'/key/'execution.json'),
                'pilot_sha256':sha(OUT/'prospective'/key/'pilot/result.json'),
                'original_late_layers_or_finalnorm_calls':0,'scope_note':'Actual early-onlyownhistory stress test; first-prefix training support does not cover laterhistory.'}
            immutable(final,result)
            print('NATURAL_PROSPECTIVE_COMPLETE',key,rule.variant,scope,len(flat),result['generated_tokens'],flush=True)
    except Exception as exc:
        failure(OUT/'prospective'/key,start,exc);raise
    finally:
        if engine is not None:engine.close()
        for h in handles:h.remove()
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    parser.add_argument('--confirmation',action='store_true')
    parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args();main(args.model,args.confirmation,args.pilot_only)
